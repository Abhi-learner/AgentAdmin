from pathlib import Path
from src.dataingestion.filestatemanager import FileStateManager
from src.dataingestion.fileprocessor import FileProcessor
from src.logger.logger import  Logger
import os

logging = Logger.get_logger(__name__)

class InitialScan:
    def __init__(self, folder_path):
        self.folder_path = folder_path


    def initial_scan(self):
        file_state = FileStateManager()
        logging.info("Running initial scan...")
        files_hash_json = file_state.load_state(self.folder_path)

        folder = Path(self.folder_path)

        for filename in os.listdir(folder):
            full_path = folder / filename
            if full_path.is_file():
                # print(full_path)
                processed_file = FileProcessor(full_path, files_hash_json)
                processed_file.process_file()