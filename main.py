from src.dataingestion.startfilewatcher import StartWatcher
from src.dataingestion.initialscan import InitialScan
from src.dataingestion.watcher import FlowTriggerHandler
from src.dataingestion.fileconfig import Config
from watchdog.observers import Observer
from src.logger.logger import Logger
import time

if __name__ == '__main__':
    watcher = StartWatcher()
    watcher.start()
