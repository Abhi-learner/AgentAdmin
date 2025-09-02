import re
import json
from typing import Any, Dict, List, Optional
from src.state.emailstate import EmailState


class IntentPolicyNormalizer:
    ALERT_MAP = {
        "disk": ("disk_cleanup", "disk_alert_subgraph"),
        "filesystem": ("disk_cleanup", "disk_alert_subgraph"),
        "cpu": ("cpu_resolution", "cpu_alert_subgraph"),
        "memory": ("memory_resolution", "memory_alert_subgraph"),
        "ram": ("memory_resolution", "memory_alert_subgraph"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            try:
                self.config = self._load_config("intent_config.json")
            except FileNotFoundError:
                self.config = {}

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _lower(s: Optional[str]) -> str:
        return (s or "").lower()

    @staticmethod
    def _collect_text(entities: Dict[str, List[Dict[str, Any]]],
                      buckets=("alerts", "events", "requests")) -> str:
        parts = []
        for b in buckets:
            for it in entities.get(b, []) or []:
                for k in ("value", "event", "request"):
                    if it.get(k):
                        parts.append(str(it[k]))
        return " | ".join(parts).lower()

    def normalize_decision(self, state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        ents = state.get("entities") or {}
        text_blob = self._collect_text(ents)
        has_ticket = bool(ents.get("tickets"))
        trace = []

        # 1. Explicit request takes precedence
        if ents.get("requests"):
            decision.update({
                "intent": "fs_extension",
                "action": "route_subgraph",
                "route": "fs_extension_subgraph",
                "requires_approval": True,
            })
            decision.setdefault("reasons", []).append("explicit request overrides alert/ticket")

        # 2. Alerts resolution
        elif ents.get("alerts") or ents.get("events"):
            matched = None
            for kw, (intent, route) in self.ALERT_MAP.items():
                if kw in text_blob:
                    matched = (intent, route)
                    break
            if matched:
                intent, route = matched
            else:
                # 🔑 Default any alert → generic "alert_resolution"
                intent, route = "alert_resolution", "generic_alert_subgraph"

            decision.update({
                "intent": intent,
                "action": "route_subgraph" if route else "manual_review",
                "route": route,
                "requires_approval": False,
            })
            decision.setdefault("reasons", []).append(f"alert triggered {intent}")

        # 3. Ticket-only updates
        elif has_ticket:
            decision.update({
                "intent": "ticket_update",
                "action": "route_subgraph",
                "route": "ticket_update_subgraph",
                "requires_approval": False,
            })
            decision.setdefault("reasons", []).append("ticket-only update chosen")

        # 4. Unclear
        else:
            decision.update({
                "intent": "unclear",
                "action": "manual_review",
                "route": None,
            })
            decision.setdefault("reasons", []).append("no clear signal, defaulted to unclear")

        # defaults
        decision.setdefault("confidence", 80)
        decision.setdefault("missing", [])
        decision.setdefault("debug", {})
        decision["debug"]["text_blob"] = text_blob
        decision["debug"]["normalizer_trace"] = trace

        return decision