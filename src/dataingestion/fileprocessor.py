from pathlib import Path
from src.dataingestion.filestatemanager import FileStateManager
from src.logger.logger import  Logger
from datetime import datetime
import extract_msg
from src.state.emailstate import EmailState
from src.emailprocessor.emailprocessor import EmailProcessor

logging = Logger.get_logger(__name__)

class FileProcessor:

    def __init__(self,path: Path,files_hash_json:dict):
        self.path = Path(path)
        self.files_hash_json = files_hash_json

    def check_file_type(self, file_info:dict):
        state = file_info
        print(state)
        # file_path = state.get("file_path", str(self.path / state["file_name"]))
        ext = Path(state["file_name"]).suffix.lower()
        if ext in [".xls", ".xlsx", ".pdf", ".txt"]:
            print("document")
            return "documents_parser"
        elif ext == ".msg":
            logging.info("Found Email")
            with open(state["source_path"], "r", encoding="utf-8") as f:
                email_text = f.read()
            email_state = EmailState({"email_text": email_text})
            email_processor = EmailProcessor(email_state)
            email_processor.start_email_processing()
            return None
            # msg = extract_msg.Message(state["source_path"])
            # subject = msg.subject or ""
            # body = msg.body or ""
            # email_text = f"{subject}\n{body}"
            # email_state = EmailState({"email_text": email_text})
            # email_processor = EmailProcessor(email_state)
            # email_processor.start_email_processing()
            return None
        else:
            return "documents_parser"

    def process_file(self) :
        file_state = FileStateManager()
        # print(f"path from processor {self.path}")
        # print(f"json from process {self.files_hash_json}")
        file_hash = file_state.compute_file_hash(self.path)
        last_modified = self.path.stat().st_mtime
        filename = self.path.name
        # print(file_hash,last_modified,filename )

        record = self.files_hash_json["files"].get(filename)
        if record is None or record["hash"] != file_hash:
            logging.info(f"Change detected in: {filename}")
            file_info = {
                "source_path": str(self.path),
                "file_name": filename,
                "timestamp": datetime.now()
            }
            self.check_file_type(file_info)
            self.files_hash_json["files"][filename] = {
                "hash": file_hash,
                "last_modified": last_modified
            }

            file_state.save_state(self.files_hash_json)
            return True
        return False



