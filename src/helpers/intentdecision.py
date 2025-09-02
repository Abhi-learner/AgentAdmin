
from typing import List, Optional, Literal
from typing import TypedDict, Literal, Optional, List

class IntentDecision(TypedDict, total=False):
    intent: str
    action: Literal["route_subgraph", "manual_review", "end"]
    route: Optional[str]
    confidence: int
    reasons: List[str]
    missing: List[str]
    requires_approval: Optional[bool]

