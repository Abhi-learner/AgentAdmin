from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.logger.logger import Logger
from src.dataingestion.filestatemanager import FileStateManager
from src.dataingestion.fileprocessor import FileProcessor
from pathlib import Path
import time

log = Logger.get_logger(__name__)

class FlowTriggerHandler(FileSystemEventHandler):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.files_hash_json = FileStateManager().load_state(self.folder_path)

    def on_created(self, event):
        if not event.is_directory:
            log.info(f"New file detected: {event.src_path}")
            time.sleep(2)
            FileProcessor(Path(event.src_path), self.files_hash_json).process_file()

    def on_modified(self, event):
        if not event.is_directory:
            log.info(f"File modified: {event.src_path}")
            FileProcessor(event.src_path, self.files_hash_json).process_file()