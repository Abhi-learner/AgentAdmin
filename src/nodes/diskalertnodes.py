import json
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
from src.helpers.diskalertextractor import DiskAlertExtractor
from src.state.diskalertstate import DiskAlertState
from src.helpers.findlargefiles import FindLargeFiles
from src.helpers.deletelargefiles import DeleteLargeFiles

logging = Logger.get_logger(__name__)

class DiskAlertNodes:
    """
    Collection of nodes used inside the Disk Alert subgraph.
    Each method is a node.
    """

    def extract_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from entities + email text.
        """
        entities = state.get("entities", {})
        email_text = state.get("email_text", "")

        extractor = DiskAlertExtractor(entities, email_text)
        disk_info = extractor.extract()

        state["disk_alert_info"] = disk_info
        logging.info("Extracted disk alert info:", disk_info)
        print("Extracted disk alert info:", disk_info)

        return state

    def discover_files(self, state: Dict[str, Any]) -> Dict[str, Any]:
        disk_info = state.get("disk_alert_info", {})
        server_name = disk_info.get("server")
        fs = disk_info.get("filesystem", "/var/log")

        if not server_name:
            raise ValueError("No server found in state['disk_alert_info']")
        conn_obj = FindLargeFiles()

        command = (
            f"find {fs} -type f "
            f"-not -path '/proc/*' "
            f"-not -path '/usr/*' "
            f"-not -path '/etc/*' "
            f"-not -path '/var/lib/*' "
            f"-not -path '/boot*' "
            f"-not -path '/sys/*' "
            f"-not -path '/var/cache/*' "
            "-printf '%s %p\\n' 2>/dev/null | sort -nr | head -n 5"
        )

        print(f"📡 Running remote discovery on {server_name} ")
        output = conn_obj.run_remote_command(server_name, command)

        files: List[Dict[str, Any]] = []
        for line in output.strip().splitlines():
            try:
                size_str, path = line.split(" ", 1)  # first part = size, second = path
                size_bytes = int(size_str)

                # Convert size into human-readable
                if size_bytes >= 1024 ** 3:
                    size_human = f"{size_bytes / 1024 ** 3:.2f}G"
                elif size_bytes >= 1024 ** 2:
                    size_human = f"{size_bytes / 1024 ** 2:.2f}M"
                elif size_bytes >= 1024:
                    size_human = f"{size_bytes / 1024:.2f}K"
                else:
                    size_human = f"{size_bytes}B"

                files.append({
                    "name": path,
                    "size_bytes": size_bytes,  # keep raw for numeric comparisons
                    "size": size_human  # human-readable
                })
            except ValueError:
                continue

        state["discovered_files"] = files
        print("📂 Discovered large files:", files)
        return state

    def cleanup_planner_node(self,state: Dict[str, Any]) -> Dict[str, Any]:
        disk_info = state.get("disk_alert_info", {})
        env = disk_info.get("env", "test").lower()
        discovered_files = state.get("discovered_files", [])

        if not discovered_files:
            state["cleanup_plan"] = {
                "action": "none",
                "files": [],
                "reason": "No large files discovered."
            }
            return state

        if env == "prod":
            state["cleanup_plan"] = {
                "action": "request_approval",
                "files": discovered_files,
                "reason": "Prod server – approval required before cleanup."
            }
        else:
            state["cleanup_plan"] = {
                "action": "execute",
                "files": discovered_files,
                "reason": f"{env.title()} server – auto-execution allowed."
            }

        return state

    def approval_node(self,state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles approval for cleanup in Prod environment.
        Expects state['cleanup_plan']['action'] == 'request_approval'
        """
        plan = state.get("cleanup_plan", {})

        # In real system: replace this with UI, API, or human-in-the-loop
        user_input = input(f"Approval needed to delete {len(plan.get('files', []))} files. Approve? (yes/no): ")

        if user_input.strip().lower() in ["yes", "y"]:
            state["approval_result"] = "approved"
            state["next_action"] = "execution_node"  # forward to execution node
        else:
            state["approval_result"] = "rejected"
            state["next_action"] = "end"  # stop flow

        return state

    def execution_node(self,state: Dict[str, Any]) -> Dict[str, Any]:
        disk_info = state.get("disk_alert_info", {})
        server_name = disk_info.get("server")
        plan = state.get("cleanup_plan", {})
        files = plan.get("files", [])
        file_paths = " ".join(f["name"] for f in files)


        if not server_name:
            raise ValueError("No server found in state['disk_alert_info']")
        conn_obj = DeleteLargeFiles()
        command = (
            f"rm -f {file_paths} "
        )
        print(f"deleting files {files} on {server_name} ")
        state = conn_obj.run_remote_command(server_name, command, state)
        return state

