from dataclasses import dataclass, field
from typing import TypedDict, List, Any
from typing import Dict



class EmailState(TypedDict, total=False):
    email_text: str
    classification: Dict[str, Any]  # {"label": "Task", "confidence": 92}
    entities: Dict[str, Dict[str, Any]]  # {"server": {"value": "db01", "confidence": 95}}
    retrieved_policies: List[Dict[str, Any]]  # [{"policy": "FS Extension Approval", "similarity": 0.89}]
    policy_eval: Dict[str, Any]  # {"requires_approval": True, "confidence": 87}
    final_decision: Dict[str, Any]  # {"action": "request_approval", "confidence": 82}

