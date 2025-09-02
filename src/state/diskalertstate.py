from typing import TypedDict, Dict, Any, List
from src.state.emailstate import EmailState

class DiskAlertState(EmailState, total=False):
    # Extra fields only visible inside this subgraph
    disk_alert_info: Dict[str, Any]
    discovered_files: List[Dict[str, Any]]
    cleanup_plan: Dict[str, Any]
    approval_result: str
    execution_result: Dict[str, Any]