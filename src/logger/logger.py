import logging
from logging.handlers import RotatingFileHandler
import os


class Logger:
    _loggers = {}
    DEFAULT_LOG_FILE = "logs/system.log"   # ✅ single log file for whole system

    @staticmethod
    def get_logger(name: str,
                   log_file: str = None,
                   level: int = logging.INFO) -> logging.Logger:
        """
        Get a logger instance that can be reused across modules/classes.
        :param name: Name of the logger (usually __name__ of module)
        :param log_file: Optional log file override (defaults to DEFAULT_LOG_FILE)
        :param level: Logging level
        :return: Logger object
        """

        if log_file is None:
            log_file = Logger.DEFAULT_LOG_FILE

        if name in Logger._loggers:
            return Logger._loggers[name]

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # prevent duplicate logs

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (rotates at 5MB, keeps 5 backups)
        fh = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        Logger._loggers[name] = logger
        return logger
