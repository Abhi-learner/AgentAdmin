from src.state.emailstate import EmailState
import json
from src.state.diskalertstate import DiskAlertState
from src.state.emailstate import EmailState
from src.logger.logger import Logger
from src.prompts.emailprompts import EmailPrompt
from src.llms.groqllm import GroqLLM
import pandas as pd
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import os
import re
from src.llms.openaillm import OpenAILLM
from typing import Dict, Any, List, Tuple
import chromadb
from langchain_openai import OpenAI
from pathlib import Path
from pandas import DataFrame
from src.helpers.intentdecision import IntentDecision
from src.helpers.intentnormalizer import IntentPolicyNormalizer
from src.workflows.diskalertworkflow import DiskGraph
from src.helpers.printstate import StatePrinter



logging = Logger.get_logger(__name__)


class EmailNodes:



    def classify_email(self,state:EmailState):
        state = state
        llm_wrapper = GroqLLM("llama-3.1-8b-instant")
        llm = llm_wrapper.get_llm()
        prompt_obj = EmailPrompt()
        prompt = prompt_obj.create_classification_prompt()
        chain = prompt | llm
        result = chain.invoke({"email_text": state["email_text"]})
        if hasattr(result, "content"):  # LangChain 0.1x / 1.x
            result_text = result.content
        else:
            result_text = str(result)  # fallback
        try:
            parsed_result = json.loads(result_text)
        except json.decoder.JSONDecodeError:
            parsed_result = {
                "classification": "Unknown",
                "confidence": 0
            }
        return {"classification": parsed_result}

    def email_entity_extraction(self, state:EmailState):
        state = state
        llm_wrapper = OpenAILLM("gpt-4o-mini")
        llm = llm_wrapper.get_llm()
        load_dotenv()
        inventory_file = os.getenv("INVENTORY")
        df = pd.read_excel(inventory_file)
        server_inventory = df["SERVER"].dropna().astype(str).str.strip().str.upper().tolist()
        fuzzy_threshold = 80
        words = re.findall(r'\b\w+\b', state["email_text"])
        matched_servers = set()
        for word in words:
            word = str(word).strip().upper()
            match, score, _ = process.extractOne(word, server_inventory, scorer=fuzz.ratio)
            if score >= fuzzy_threshold:
                matched_servers.add(match)

        candidate_servers = list(matched_servers)
        candidate_tickets = re.findall(r"\b(?:WO|INC|CRQ)\d+\b", state["email_text"], re.IGNORECASE)
        alert_keywords = [
            "error", "failed", "failure", "unavailable", "not responding",
            "down", "crash", "timeout", "unreachable", "restart", "warning",
            "high cpu", "cpu usage", "memory usage", "disk space", "disk full",
            "latency", "slow", "packet loss", "critical", "alarm", "overload"
        ]
        text_lower = state["email_text"].lower()
        candidates = []
        for keyword in alert_keywords:
            if keyword in text_lower:
                candidates.append(keyword)

        # Optionally also extract noun phrases around keywords (simple regex)
        extractions = re.findall(r'(\w+\s+(error|failure|issue|warning))', text_lower)
        for phrase, _ in extractions:
            if phrase not in candidates:
                candidates.append(phrase)

        # Remove duplicates while preserving order
        candidate_alerts = list(dict.fromkeys(candidates))
        prompt_obj = EmailPrompt()
        prompt = prompt_obj.create_entities_extraction_prompt()
        chain = prompt | llm
        result = chain.invoke({"candidate_servers": candidate_servers, "candidate_tickets": candidate_tickets, "email_text": state["email_text"], "candidate_alerts": candidate_alerts })
        try:
            entities = json.loads(result.content)
        except json.decoder.JSONDecodeError:
            entities = {
                "servers": [{"value": s, "confidence": 95} for s in candidate_servers],
                "tickets": [{"value": t, "confidence": 90} for t in candidate_tickets],
                "requests": [],
                "alerts": [],
                "events": [],
                "links": []
            }
        df["SERVER"] = df["SERVER"].astype(str).str.strip().str.lower()
        df["Env"] = df["Env"].astype(str).str.strip()
        server_lookup = dict(zip(df["SERVER"], df["Env"]))
        for server in entities["servers"]:
            server_name = server.get("value", "").strip().lower()
            server_env = server_lookup.get(server_name, "")
            server["Env"] = server_env

        logging.info(f"Entities: {entities}")


        return {"entities": entities}

    def policy_extraction(self, state:EmailState):
        load_dotenv()
        policy_file_path = os.getenv("POLICY_FILE")
        """
            Retrieve policies from JSON/dict based on entities in state.
            Handles both JSON string and dict safely.
            """
        if not policy_file_path:
            raise ValueError("POLICY_FILE not set in .env")

        with open(policy_file_path, "r") as f:
            policy_db = json.load(f)

        if isinstance(policy_db, str):
            try:
                policy_db = json.loads(policy_db.strip())
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in POLICY_FILE: {e}")

        if not isinstance(policy_db, dict):
            raise TypeError("policy_db must be a dict or valid JSON string")

            # ---- Extract entities from state ----
        entities = state.get("entities", {})

        server_env = ""
        servers = entities.get("servers", [])
        if servers and "Env" in servers[0]:
            server_env = servers[0]["Env"]

        action = ""
        alerts = entities.get("alerts", [])
        if alerts:
            action = alerts[0]["value"]

        trigger = ""
        events = entities.get("events", [])
        if events:
            trigger = events[0]["value"]

        print(f"[DEBUG] Extracted entities → server_env={server_env}, action={action}, trigger={trigger}")

        # ---- Policy matching ----
        retrieved_policies = []

        for env_key, actions in policy_db.items():
            env_score = fuzz.partial_ratio(server_env.lower(), env_key.lower())

            for act_key, policy in actions.items():
                action_score = fuzz.partial_ratio(action.lower(), act_key.lower())
                trigger_score = 0

                if "condition" in policy and "trigger" in policy["condition"]:
                    trigger_score = fuzz.partial_ratio(
                        trigger.lower(),
                        policy["condition"]["trigger"].lower()
                    )

                similarity = round(
                    (0.4 * env_score + 0.5 * action_score + 0.1 * trigger_score) / 100, 2
                )

                print(
                    f"[DEBUG] Comparing env={env_key}, action={act_key}, scores → env={env_score}, action={action_score}, trigger={trigger_score}, similarity={similarity}")

                if similarity > 0.6:
                    retrieved_policies.append({
                        "policy": policy.get("policy", f"{act_key.title()} Policy"),
                        "requires_approval": policy.get("requires_approval", False),
                        "approval_source": policy.get("approval_source"),
                        "similarity": similarity
                    })

        return {"retrieved_policies": retrieved_policies}

    # def policy_eval_node(self, state: EmailState) :
    #     """
    #     Calls LLM with policy-eval prompt and writes:
    #     state['policy_eval'] = {
    #       'requires_approval': bool,
    #       'decision': 'execute'|'request_approval'|'deny',
    #       'recommended_action': str|None,
    #       'parameters': dict,
    #       'confidence': int
    #     }
    #     """
    #
    #     def _max_sim(retrieved_policies: List[Dict[str, Any]]) -> int:
    #         sims = [p.get("similarity", 0) for p in (retrieved_policies or [])]
    #         return int(max(sims or [0]) * 100)
    #
    #     def _fallback() -> Dict[str, Any]:
    #         # conservative fallback: prod + risky → approval; else execute low confidence
    #         entities = state.get("entities", {}) or {}
    #         env = ""
    #         servers = entities.get("servers", [])
    #         if servers:
    #             env = (servers[0].get("Env") or servers[0].get("env") or "").lower()
    #         action_hint = ""
    #         if entities.get("requests"):
    #             action_hint = entities["requests"][0].get("value", "").lower()
    #         elif entities.get("alerts"):
    #             action_hint = entities["alerts"][0].get("value", "").lower()
    #         elif entities.get("events"):
    #             action_hint = entities["events"][0].get("value", "").lower()
    #         risky = {"file_deletion", "backup_restore", "patch_installation", "restart_service", "filesystem_extension"}
    #         if env == "prod" and any(r in action_hint.replace(" ", "_") for r in risky):
    #             return {"requires_approval": True, "decision": "request_approval", "recommended_action": None,
    #                     "parameters": {}, "confidence": 75}
    #         return {"requires_approval": False, "decision": "execute", "recommended_action": action_hint or None,
    #                 "parameters": {}, "confidence": 60}
    #     prompt_obj = EmailPrompt()
    #     prompt = prompt_obj.make_policy_eval_prompt(state)
    #
    #     try:
    #         llm_wrapper = GroqLLM("llama-3.1-8b-instant")
    #         llm = llm_wrapper.get_llm()
    #         raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    #
    #         # normalize to dict
    #         if isinstance(raw, dict):
    #             parsed = raw
    #         else:
    #             text = str(raw).strip()
    #             # extract JSON portion if needed
    #             i, j = text.find("{"), text.rfind("}")
    #             if i != -1 and j != -1 and j > i:
    #                 text = text[i:j + 1]
    #             parsed = json.loads(text)
    #
    #         decision = parsed.get("decision")
    #         requires = bool(parsed.get("requires_approval"))
    #         action = parsed.get("recommended_action")
    #         params = parsed.get("parameters") or {}
    #         conf = int(parsed.get("confidence", 0))
    #         conf = max(0, min(100, conf))
    #
    #         # nudge confidence with vector similarity prior
    #         prior = _max_sim(state.get("retrieved_policies", []))
    #         if prior:
    #             conf = int(0.7 * conf + 0.3 * prior)
    #
    #         # reconcile decision ↔ requires_approval
    #         if decision == "execute":
    #             requires = False
    #         elif decision == "request_approval":
    #             requires = True
    #
    #         state["policy_eval"] = {
    #             "requires_approval": requires,
    #             "decision": decision or ("request_approval" if requires else "execute"),
    #             "recommended_action": action,
    #             "parameters": params,
    #             "confidence": conf
    #         }
    #         return state
    #
    #     except Exception as e:
    #         # robust fallback: keep graph moving
    #         fb = _fallback()
    #         state["policy_eval"] = fb
    #         state.setdefault("_errors", []).append({"node": "policy_eval", "error": str(e), "fallback_used": True})
    #         return state
    def policy_eval_node(self, state: EmailState):
        """Read Excel/Chroma if present, build prompt, call LLM, return updated state.
        This function does NOT write to Excel/Chroma.
        """
        load_dotenv()
        excel_path = os.getenv("MEMORY_DB")
        chroma_dir = os.getenv("MEMORY_VECTOR_DB")
        CASES_COLLECTION = os.getenv("MEMORY_VECTOR_DB_NAME")
        openai_client = OpenAI()
        llm_wrapper = GroqLLM("llama-3.1-8b-instant")
        llm = llm_wrapper.get_llm()
        # Ensure structure
        # state.setdefault("classification", {})
        # state.setdefault("entities", {})
        # state.setdefault("retrieved_policies", [])

        # ----- Inline helpers (kept inside to honor "only two functions") -----
        def _load_excel_if_exists(path_str: str) -> Tuple[DataFrame, DataFrame]:
            path = Path(path_str)
            if not path.exists():
                return pd.DataFrame(), pd.DataFrame()
            try:
                xls = pd.ExcelFile(path)
                sheets = {name.lower(): name for name in xls.sheet_names}
                inv = xls.parse(sheets.get("inventory", xls.sheet_names[0])) if xls.sheet_names else pd.DataFrame()
                ev = xls.parse(sheets.get("events", xls.sheet_names[-1])) if xls.sheet_names else pd.DataFrame()
                return inv, ev
            except Exception:
                return pd.DataFrame(), pd.DataFrame()

        def _fallback_embed(text: str) -> List[float]:
            h = abs(hash(text))
            return [((h >> (i % 32)) & 0xFF) / 255.0 for i in range(256)]

        def _read_chroma_similar_cases(chroma_dir: str, query_vec: List[float], n: int = 4) -> List[str]:
            path = Path(chroma_dir)
            if not path.exists():
                return []
            try:
                client = chromadb.PersistentClient(path=chroma_dir)
                names = [c.name for c in client.list_collections()]
                if CASES_COLLECTION not in names:
                    return []
                col = client.get_collection(CASES_COLLECTION)
                res = col.query(query_embeddings=[query_vec], n_results=n)
                out: List[str] = []
                if res and res.get("documents"):
                    for doc in res["documents"][0]:
                        if doc:
                            out.append(doc[:400])
                return out
            except Exception:
                return []

        # ----- Read Excel (if present) -----
        inv_df, ev_df = _load_excel_if_exists(excel_path)

        entities = state.get("entities", {}) or {}
        server = (entities.get("server", {}) or {}).get("value", "")
        env = (entities.get("server", {}) or {}).get("env") or (entities.get("environment", {}) or {}).get("value", "")

        # inventory row
        inventory: Dict[str, Any] = {}
        if not inv_df.empty and server and "server_name" in inv_df.columns:
            match = inv_df[inv_df["server_name"].astype(str).str.lower() == server.lower()]
            if not match.empty:
                inventory = match.iloc[0].to_dict()
                if not env:
                    env = str(inventory.get("environment", ""))

        # recent events
        recent_events: List[Dict[str, Any]] = []
        if not ev_df.empty and server and "server_name" in ev_df.columns:
            sub = ev_df[ev_df["server_name"].astype(str).str.lower() == server.lower()]
            if not sub.empty:
                recent_events = sub.tail(5).to_dict(orient="records")

        # similar cases from Chroma (read-only)
        email_text: str = state.get("email_text") or ""
        if openai_client is not None:
            try:
                resp = openai_client.embeddings.create(model="text-embedding-3-small", input=[email_text])
                query_vec = resp.data[0].embedding
            except Exception:
                query_vec = _fallback_embed(email_text)
        else:
            query_vec = _fallback_embed(email_text)
        similar_cases = _read_chroma_similar_cases(chroma_dir, query_vec, n=4)

        # policies from state
        def _norm_env(x: str) -> str:
            x = (x or "").strip().lower()
            aliases = {
                "prod": "prod", "production": "prod", "live": "prod",
                "test": "test", "qa": "test", "uat": "test", "staging": "test",
                "pet": "pet", "preprod": "test", "dev": "test"
            }
            return aliases.get(x, x)

        env = _norm_env(state.get("entities", {}).get("server", {}).get("env")
                        or state.get("entities", {}).get("environment", {}).get("value"))

        retrieved_policies = state.get("retrieved_policies", []) or []

        def _score(p: dict) -> tuple:
            env_match = 1 if _norm_env(p.get("environment")) == env else 0
            sim = p.get("similarity", 0) or 0
            return (env_match, sim)
        # retrieved_policies = state.get("retrieved_policies", []) or []
        # top_policies = sorted(retrieved_policies, key=lambda p: p.get("similarity", 0), reverse=True)[:5]
        top_policies = sorted(retrieved_policies, key=_score, reverse=True)[:2]

        # Build prompt
        prompt_obj = EmailPrompt()
        ptxt = prompt_obj.make_policy_eval_prompt(
            state,
            inventory=inventory,
            recent_events=recent_events,
            similar_cases=similar_cases,
            top_policies=top_policies,
        )

        # Call LLM & parse
        try:
            raw = llm.invoke(ptxt) if hasattr(llm, "invoke") else llm(ptxt)
            if isinstance(raw, dict):
                parsed = raw
            else:
                text = str(raw).strip()
                i, j = text.find("{"), text.rfind("}")
                if i != -1 and j != -1 and j > i:
                    text = text[i:j + 1]
                parsed = json.loads(text)

            decision = (parsed.get("decision") or "").lower()
            requires = bool(parsed.get("requires_approval"))
            rec_act = parsed.get("recommended_action") or canonical_action
            params = parsed.get("parameters") or {}
            conf = int(parsed.get("confidence", 0))
            conf = max(0, min(100, conf))

            # Confidence calibration via best policy similarity
            max_sim = int(max([p.get("similarity", 0) for p in top_policies] or [0]) * 100)
            if max_sim:
                conf = int(0.7 * conf + 0.3 * max_sim)

            if decision == "execute":
                requires = False
            elif decision == "request_approval":
                requires = True

            state["policy_eval"] = {
                "requires_approval": requires,
                "decision": decision or ("request_approval" if requires else "execute"),
                "recommended_action": rec_act,
                "parameters": params,
                "confidence": conf,
            }

        except Exception as e:
            state["policy_eval"] = {
                "requires_approval": True,
                "decision": "request_approval",
                "recommended_action": None,
                "parameters": {},
                "confidence": 60,
            }
            state.setdefault("_errors", []).append({"node": "policy_eval", "error": str(e), "fallback_used": True})

        return state

    def decision_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads state['policy_eval'] and writes state['final_decision'].
        Output actions:
          - 'request_approval' → go to approval node
          - 'tool'             → go to tool node
          - 'manual_review'    → escalate to a human triage node
        """
        load_dotenv()
        HIGH_CONF = os.getenv("HIGH_CONF")
        MID_CONF = os.getenv("MID_CONF")

        pe = state.get("policy_eval", {}) or {}
        decision = (pe.get("decision") or "").lower()  # 'execute' | 'request_approval' | 'deny'
        requires = bool(pe.get("requires_approval"))
        conf = int(pe.get("confidence") or 0)

        # Default
        action = "manual_review"

        # High confidence path: trust the LLM decision
        if conf >= HIGH_CONF:
            if decision == "request_approval" or requires:
                action = "request_approval"
            elif decision == "execute":
                action = "tool"
            elif decision == "deny":
                action = "manual_review"  # or short-circuit to persistence/log
            else:
                action = "manual_review"

        # Medium confidence path: be conservative
        elif MID_CONF <= conf < HIGH_CONF:
            # If LLM says approval → ask approval
            if decision == "request_approval" or requires:
                action = "request_approval"
            # If LLM says execute → ask lightweight approval (treat as approval)
            elif decision == "execute":
                action = "request_approval"
            else:
                action = "manual_review"

        # Low confidence path: manual review
        else:
            action = "manual_review"

        state["final_decision"] = {
            "action": action,  # 'request_approval' | 'tool' | 'manual_review'
            "decision": decision or None,  # keep original LLM decision for audit
            "requires_approval": requires,
            "confidence": conf,
        }
        return state

    def manual_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hand-off to human triage; you can:
        - create a ticket
        - post to Slack/Teams
        - write to a review queue
        """
        # placeholder: mark that we routed to manual
        state["approval"] = "manual_review"
        return state

    def decision_intent_node(self, state: EmailState):
        import os, json, logging
        MIN_ROUTE_CONFIDENCE = 70
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), "intent_config.json")

        # init once per call (or move to module-level singleton if you prefer)
        normalizer = IntentPolicyNormalizer(config_path=CONFIG_PATH)

        # build + run LLM
        prompt_obj = EmailPrompt()
        prompt = prompt_obj.build_decision_prompt()
        llm = GroqLLM("llama-3.1-8b-instant", temperature=0).get_llm(response_format={"type": "json_object"})
        chain = prompt | llm

        raw = chain.invoke({
            "classification": state.get("classification") or {},
            "entities": state.get("entities") or {},
            "policies": state.get("retrieved_policies") or [],
            "policy_eval": state.get("policy_eval") or {},
        })

        # extract text
        text = getattr(raw, "content", raw)

        # parse LLM JSON (fallback on failure)
        try:
            decision = json.loads(text)
        except Exception:
            decision = {
                "intent": "unclear",
                "action": "manual_review",
                "route": None,
                "confidence": 0,
                "reasons": ["Failed to parse JSON"],
                "missing": [],
                "requires_approval": None,
            }

        # 🔹 ALWAYS normalize (was previously only in except branch)
        decision = normalizer.normalize_decision(state, decision)

        # make sure reasons exists before appending later
        decision.setdefault("reasons", [])

        # optional: bump confidence if policy matched a actionable route
        if decision.get("intent") and decision.get("action") == "route_subgraph":
            decision["confidence"] = max(decision.get("confidence", 0), 90)

        # your confidence gate
        if decision.get("action") == "route_subgraph" and decision.get("confidence", 0) < MIN_ROUTE_CONFIDENCE:
            decision["action"] = "manual_review"
            decision["route"] = None
            decision["reasons"].append(f"Confidence {decision.get('confidence')} < {MIN_ROUTE_CONFIDENCE}")

        logging.info({"final_decision": decision})

        # LangGraph: return only the delta
        return {"final_decision": decision}

    def route_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide where to route next based on final_decision.
        """
        decision = state.get("final_decision", {})
        action = decision.get("action")
        route = decision.get("route")

        next_node = None
        if action == "route_subgraph" and route:
            next_node = route
        elif action == "manual_review":
            next_node = "manual_review"
        else:
            next_node = "end"  # fallback

        return {"next_node": next_node}

    def manual_review(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder for manual review.
        In real system, this might notify a human or log for triage.
        """
        logging.info("Sending email to manual review:", state.get("email_text"))
        return state

    def handle_disk_alert(self, disk_alert_state:EmailState):
        state = disk_alert_state
        logging.info(" Handling disk alert:", disk_alert_state.get("entities"))
        graph = DiskGraph()
        graph_builder = graph.create_disk_alert_graph()
        compiled_graph = graph_builder.compile()
        result = compiled_graph.invoke(state)
        printer = StatePrinter(result)
        printer.print_state()

        return state

    def handle_cpu_alert(self, state):
        logging.info("Handling CPU alert:", state.get("entities"))
        return state

    def handle_fs_extension(self, state):
        logging.info(" Handling filesystem extension request:", state.get("entities"))
        return state

    def handle_ticket_update(self, state):
        logging.info(" Handling ticket update:", state.get("entities"))
        return state
    def handle_memory_alert(self, state):
        logging.info(" Handling Memory Alert:", state.get("entities"))
        return state

















