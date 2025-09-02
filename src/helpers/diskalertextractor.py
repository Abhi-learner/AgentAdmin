import re
from typing import Dict, Any, Optional


class DiskAlertExtractor:
    """
    Extract structured information from disk utilization alerts.
    """

    def __init__(self, entities: Dict[str, Any], email_text: str):
        self.entities = entities or {}
        self.email_text = email_text or ""

    def _extract_server(self) -> (Optional[str], Optional[str]):
        server = None
        env = None
        if self.entities.get("servers"):
            server = self.entities["servers"][0].get("value")
            env = self.entities["servers"][0].get("Env")
        return server, env

    def _extract_alert(self) -> Optional[str]:
        if self.entities.get("alerts"):
            return self.entities["alerts"][0].get("value")
        return None

    def _extract_utilization(self) -> Optional[int]:
        for e in self.entities.get("events", []):
            m = re.search(r"(\d{1,3})\s*%", e.get("value", ""))
            if m:
                return int(m.group(1))
        return None

    def _extract_filesystem(self) -> Optional[str]:
        # Match "/" OR "/something" OR "/something/more"
        fs_match = re.search(r"(/[\w/]*)", self.email_text)
        if fs_match:
            return fs_match.group(1)
        return None

    def extract(self) -> Dict[str, Optional[Any]]:
        """
        Public method to extract structured disk alert information.

        Returns:
            {
                "server": str | None,
                "env": str | None,
                "alert": str | None,
                "filesystem": str | None,
                "utilization_percent": int | None
            }
        """
        server, env = self._extract_server()
        return {
            "server": server,
            "env": env,
            "alert": self._extract_alert(),
            "filesystem": self._extract_filesystem(),
            "utilization_percent": self._extract_utilization(),
        }
