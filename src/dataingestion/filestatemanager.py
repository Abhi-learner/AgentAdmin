import logging
import os
import json
import time
import hashlib
from src.logger.logger import Logger
from src.dataingestion.fileconfig import Config
from pathlib import Path


log = Logger.get_logger(__name__)

class FileStateManager:
    log = Logger.get_logger(__name__)
    def __init__(self):
        self.config = Config()
        self.STATE_FILE = self.config.get_state_file()

    def compute_file_hash(self,file_path: Path) -> str:
        file_path = file_path.resolve()
        # print(file_path.exists())
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_state(self, files_list: dict) -> None:
        with open(self.STATE_FILE, "w") as f:
            json.dump(files_list, f, indent=4)

    def load_state(self, folder_path: str) -> dict:
        state_path = Path(self.STATE_FILE)

        if state_path.exists():
            try:
                content = state_path.read_text().strip()
                if not content:
                    log.error(f"Empty file")
                    raise ValueError("Empty file")  # ✅ Handle empty file
                parsed = json.loads(content)
                if "folder_path" not in parsed or "files" not in parsed:
                    raise ValueError("Missing keys")
                return parsed
            except Exception as e:
                log.warning(f" Invalid or empty state file. Reinitializing. Reason: {e}")

        # ✅ Default structure
        files_list = {
            "folder_path": folder_path,
            "files": {}
        }
        self.save_state(files_list)
        return files_list

