from src.dataingestion.initialscan import InitialScan
from src.dataingestion.watcher import FlowTriggerHandler
from src.dataingestion.fileconfig import Config
from watchdog.observers import Observer
from src.logger.logger import Logger
import time

logging = Logger.get_logger(__name__)

class StartWatcher():
    def __init__(self):
        self.config = Config()
        self.watch_folder = self.config.get_watch_folder()
        print(self.watch_folder)

    def start(self):
        InitialScan(self.watch_folder).initial_scan()
        event_handler = FlowTriggerHandler(self.watch_folder)
        observer = Observer()
        observer.schedule(event_handler, path=self.watch_folder, recursive=False)
        observer.start()
        logging.info("Watching folder: %s" % self.watch_folder)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logging.info("Stopped watching.")
        observer.join()
