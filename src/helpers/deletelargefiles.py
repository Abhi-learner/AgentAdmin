import re
import os
import paramiko
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List
from src.logger.logger import Logger
from src.state.diskalertstate import DiskAlertState

logger = logging = Logger.get_logger(__name__)

class DeleteLargeFiles:
    def __init__(self):
        load_dotenv()
        inventory_path = os.getenv("INVENTORY")
        if not inventory_path:
            logger.error("No INVENTORY environment variable")
            raise ValueError("Server Inventory Path is not in .env")
        df = pd.read_excel(inventory_path)
        df.columns = [c.strip().lower() for c in df.columns]
        self.inventory = df.to_dict(orient="records")

    def _find_server_conn(self, server_name: str) -> Dict[str, Any]:
        for row in self.inventory:
            if str(row["server"]).lower() == str(server_name).lower():  # ✅ use argument
                print("DEBUG: matched row", row)
                return row
        return {}

    def run_remote_command(self, server: str, command: str, state:DiskAlertState) -> str:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        conn = self._find_server_conn(server)
        ssh.connect(
            hostname=conn["ip"],
            port=int(conn.get("port", 22)),
            username=conn["user"],
            key_filename=conn.get("key_file"),

        )
        print(f"connected to {server}, Running Command: {command}")
        deleted: List[str] = []
        plan = state.get("cleanup_plan", {})
        files = plan.get("files", [])
        file_paths = " ".join(f["name"] for f in files)
        deleted: List[str] = []
        failed: List[Dict[str, Any]] = []

        stdin, stdout, stderr = ssh.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode()
        if exit_status == 0:
            deleted.append(file_paths)
        else:

            failed.append({
                "file": file_paths,
                "error": stderr.read().decode() or f"Exit {exit_status}"
            })

        ssh.close()
        state["execution_result"] = {"deleted": deleted, "failed": failed}
        # print(state)
        return state
