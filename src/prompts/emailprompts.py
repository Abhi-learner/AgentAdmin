from langchain.prompts import PromptTemplate, ChatPromptTemplate
from src.state.emailstate import EmailState
import json
from typing import Dict, Any, List, Tuple


class EmailPrompt():



    def create_classification_prompt(self):
        CLASSIFICATION_PROMPT = """
        You are an email classifier for a system administration team. 
        Classify the email into EXACTLY one of these categories:

        - Informational
        - Alert
        - Task
        - Query

        Respond in JSON format:
        {{
          "classification": "<one of the four>",
          "confidence": <number between 0 and 100>
        }}

        ---
        Email:
        {email_text}
        ---
        """
        prompt_template = PromptTemplate(
            template=CLASSIFICATION_PROMPT,
            input_variables=["email_text"]
        )
        return prompt_template

    def create_entities_extraction_prompt(self):
        EXTRACTION_PROMPT = """
        You are an intelligent assistant specialized in IT operations and alert processing. 
        Your job is to extract structured information from incoming emails. 

        The email may contain:
        - Requests (approvals, actions, manual interventions)
        - Alerts (errors, failures, restarts, performance issues, warnings)
        - Tickets (always formatted as WO123, INC123, CRQ123, etc.)
        - Server names (validate against provided candidates)
        - Events (explicit or implicit, e.g., "error", "disk space low", "service failed to start")

        ---
        IMPORTANT RULES:
        1. If you detect keywords like ["error", "failed", "down", "unavailable", "CPU", 
           "memory", "disk", "restart", "crash", "timeout"], treat them as EVENTS or ALERTS.
        2. Always return empty arrays if nothing is found (do not omit the keys).
        3. Use provided candidate servers/tickets/alerts to validate matches. 
        4. Include confidence scores (0-100) for each entity.
        5. Link tickets to servers if contextually obvious.

        ---
        Example:

        Email:
        \"\"\"
        Monitoring detected high CPU usage on linux1. Ticket INC123 created for investigation.
        \"\"\"

        Candidate servers: ["linux1", "linux2"]
        Candidate tickets: ["INC123"]
        Candidate alerts: ["CPU usage", "disk error"]

        Extraction:
        {{
          "servers": [{{"value": "linux1", "confidence": 95}}],
          "tickets": [{{"value": "INC123", "confidence": 95}}],
          "requests": [],
          "alerts": [{{"value": "High CPU usage", "confidence": 90}}],
          "events": [{{"value": "CPU utilization exceeded threshold", "confidence": 85}}],
          "links": [{{"server": "linux1", "ticket": "INC123"}}]
        }}

        ---
        Now extract from the new email below:

        Email:
        \"\"\"
        {email_text}
        \"\"\"

        Candidate servers: {candidate_servers}
        Candidate tickets: {candidate_tickets}
        Candidate alerts: {candidate_alerts}
        """

        prompt_template = PromptTemplate(
            template=EXTRACTION_PROMPT,
            input_variables=["email_text", "candidate_servers", "candidate_tickets"]
        )
        return prompt_template

    def make_policy_eval_prompt(self,
            state: EmailState,
            inventory: Dict[str, Any],
            recent_events: List[Dict[str, Any]],
            similar_cases: List[str],
            top_policies: List[Dict[str, Any]],
    ) -> str:
        email_text = state.get("email_text", "") or ""
        classification = state.get("classification", {}) or {}
        entities = state.get("entities", {}) or {}

        server = (entities.get("server", {}) or {}).get("value", "")
        environment = (entities.get("server", {}) or {}).get("env") or (entities.get("environment", {}) or {}).get(
            "value", "")

        action_hint = (
                          (entities.get("requests", [{}])[0] or {}).get("value") if entities.get("requests") else None
                      ) or (
                          (entities.get("alerts", [{}])[0] or {}).get("value") if entities.get("alerts") else None
                      ) or (
                          (entities.get("events", [{}])[0] or {}).get("value") if entities.get("events") else None
                      ) or (
                          (entities.get("action", {}) or {}).get("value")
                      ) or ""

        # canonicalize action
        canon = (action_hint or "").lower().replace(" ", "_")
        aliases = {
            "memory_utilization_alert": "high_memory_utilization",
            "cpu_utilization_alert": "cpu_utilization",
            "disk_utilization_alert": "disk_utilization",
            "extend_filesystem": "filesystem_extension",
            "password_reset": "user_password_reset",
            "access_request": "user_access_request",
            "patch_install": "patch_installation",
        }
        for k, v in aliases.items():
            if k in canon:
                canon = v
                break
        action_hint = canon or action_hint or ""

        ctx = {
            "email_excerpt": email_text[:4000],
            "classification": classification,
            "environment": environment,
            "server": server,
            "action_hint": action_hint,
            "inventory": inventory or {},
            "recent_events": recent_events or [],
            "policies": [
                {k: p.get(k) for k in ("policy", "requires_approval", "approval_source", "similarity") if k in p}
                for p in (top_policies or [])
            ],
            "similar_past_cases": list(similar_cases or [])[:3],
        }
        system = (
            "You are a strict change-control assistant. Using ONLY this context, decide whether to "
            "execute now, request approval, or deny. Be conservative in Production. "
            "Return STRICT JSON:\n"
            '{\n'
            '  "decision": "execute" | "request_approval" | "deny",\n'
            '  "requires_approval": true|false,\n'
            '  "recommended_action": "<canonical_action_or_null>",\n'
            '  "parameters": {"server": "<name or empty>", "scope": "<e.g. /home or empty>"},\n'
            '  "confidence": <0-100>\n'
            '}\n'
            "Do not add extra text."
        )
        return f"{system}\n\nCONTEXT:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}"

    # def make_policy_eval_prompt(self,state: EmailState):

        # """
        #     Build a strict prompt for policy decision:
        #     Decide 'execute' | 'request_approval' | 'deny' for the current email.
        #     Uses: email_text, entities, retrieved_policies (with similarity), retrieved_cases (optional)
        #     """
        # email_text: str = state.get("email_text", "")
        # entities: Dict[str, Any] = state.get("entities", {}) or {}
        # retrieved: List[Dict[str, Any]] = state.get("retrieved_policies", []) or []
        # cases: List[Dict[str, Any]] = state.get("retrieved_cases", []) or []
        #
        # # extract env/action/server from your entities format
        # env = ""
        # server = ""
        # servers = entities.get("servers", [])
        # if servers:
        #     env = servers[0].get("Env") or servers[0].get("env") or ""
        #     server = servers[0].get("value") or ""
        #
        # action_hint = ""
        # if entities.get("requests"):
        #     action_hint = entities["requests"][0].get("value", "")
        # elif entities.get("alerts"):
        #     action_hint = entities["alerts"][0].get("value", "")
        # elif entities.get("events"):
        #     action_hint = entities["events"][0].get("value", "")
        #
        # # top-k policies compact
        # top_policies = sorted(retrieved, key=lambda p: p.get("similarity", 0), reverse=True)[:5]
        # for p in top_policies:
        #     # keep only essential keys to reduce tokens
        #     for k in list(p.keys()):
        #         if k not in ("policy", "requires_approval", "approval_source", "similarity"):
        #             p.pop(k, None)
        #
        # # compact case snippets
        # cases_snips = []
        # for c in cases[:3]:
        #     t = c.get("text") or ""
        #     if t:
        #         cases_snips.append(t[:400])
        #
        # system_msg = (
        #     "You are a strict change-control assistant. "
        #     "Using ONLY the provided environment, action hint, and policies, decide whether to execute now, "
        #     "request approval, or deny. Be conservative in Production. "
        #     "Return STRICT JSON with keys exactly:\n"
        #     '{\n'
        #     '  "decision": "execute" | "request_approval" | "deny",\n'
        #     '  "requires_approval": true|false,\n'
        #     '  "recommended_action": "<canonical_action_or_null>",\n'
        #     '  "parameters": {"server": "<name or empty>", "scope": "<e.g. /home or empty>"},\n'
        #     '  "confidence": <0-100>\n'
        #     "}\n"
        #     "Do not add extra text."
        # )
        #
        # ctx = {
        #     "email": email_text[:4000],
        #     "environment": env,
        #     "server": server,
        #     "action_hint": action_hint,
        #     "policies": top_policies,
        #     "similar_past_cases": cases_snips
        # }
        #
        # prompt = f"{system_msg}\n\nCONTEXT:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}"
        # return prompt

    def build_decision_prompt(self):
        from langchain.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a decision engine for an admin email triage workflow.\n"
             "You must always infer a valid INTENT and ACTION.\n\n"
             "Available INTENTS:\n"
             " - 'disk_cleanup'      → for disk or filesystem utilization alerts\n"
             " - 'cpu_resolution'    → for CPU utilization alerts\n"
             " - 'memory_resolution' → for memory/RAM utilization alerts\n"
             " - 'fs_extension'      → for requests to extend/increase disk/filesystem\n"
             " - 'ticket_update'     → for ticket-only updates (no new alert/request)\n"
             " - 'unclear'           → only if absolutely no alert, request, or ticket signal\n\n"
             "CONTROL ACTION:\n"
             " - Always 'route_subgraph' if INTENT is one of the alert or request categories\n"
             " - 'manual_review' only if intent=unclear\n\n"
             "Priority Rules (strict):\n"
             " 1. If there is any explicit REQUEST (like filesystem/disk extension), choose 'fs_extension'.\n"
             " 2. Else if ALERT present:\n"
             "    - Disk/Filesystem words → 'disk_cleanup'\n"
             "    - CPU words → 'cpu_resolution'\n"
             "    - Memory/RAM words → 'memory_resolution'\n"
             " 3. Else if TICKET present with no alert/request → 'ticket_update'.\n"
             " 4. Only use 'unclear' when nothing else applies.\n\n"
             "Respond ONLY in valid JSON with keys:\n"
             "{{\"intent\": str, "
             "\"action\": \"route_subgraph\"|\"manual_review\", "
             "\"route\": str|null, \"confidence\": int, "
             "\"reasons\": [str], \"missing\": [str], "
             "\"requires_approval\": bool|null}}."
             ),

            # Few-shot: Disk Alert → disk_cleanup
            ("user",
             'classification={{"classification":"Alert","confidence":92}}\n'
             'entities={{"alerts":[{{"value":"Disk Utilization Exceeded","confidence":92}}],'
             '"events":[{{"value":"/var at 95%","confidence":88}}]}}'
             ),
            ("assistant",
             '{{"intent":"disk_cleanup","action":"route_subgraph","route":"disk_alert_subgraph",'
             '"confidence":90,"reasons":["disk alert detected"],"missing":[],"requires_approval":false}}'
             ),

            # Few-shot: CPU Alert → cpu_resolution
            ("user",
             'classification={{"classification":"Alert","confidence":90}}\n'
             'entities={{"alerts":[{{"value":"CPU Utilization Exceeded","confidence":89}}]}}'
             ),
            ("assistant",
             '{{"intent":"cpu_resolution","action":"route_subgraph","route":"cpu_alert_subgraph",'
             '"confidence":88,"reasons":["cpu alert detected"],"missing":[],"requires_approval":false}}'
             ),

            # Few-shot: Request → fs_extension
            ("user",
             'classification={{"classification":"Request","confidence":90}}\n'
             'entities={{"requests":[{{"value":"Filesystem Extension","confidence":87}}]}}'
             ),
            ("assistant",
             '{{"intent":"fs_extension","action":"route_subgraph","route":"fs_extension_subgraph",'
             '"confidence":91,"reasons":["explicit request overrides alert"],'
             '"missing":["target filesystem/path","desired size"],"requires_approval":true}}'
             ),

            # Few-shot: Ticket update
            ("user",
             'classification={{"classification":"Alert","confidence":85}}\n'
             'entities={{"tickets":[{{"value":"INC123","confidence":90}}],'
             '"events":[{{"value":"Cleanup completed","confidence":80}}]}}'
             ),
            ("assistant",
             '{{"intent":"ticket_update","action":"route_subgraph","route":"ticket_update_subgraph",'
             '"confidence":85,"reasons":["ticket-only update chosen"],"missing":[],"requires_approval":false}}'
             ),

            # Live input (runtime variables, so single braces)
            ("user",
             "classification={classification}\nentities={entities}\npolicies={policies}\npolicy_eval={policy_eval}"
             ),
        ])


