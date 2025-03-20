import os
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    """Handles logging for errors and events using RotatingFileHandler."""
    
    loggers = {}  # Keeps track of loggers to avoid duplicates

    def __init__(self, log_filename="pipeline.log", verbose=False, maxBytes=5*1024*1024, backupCount=5):
        """
        Parameters:
        - log_filename (str): Name of the log file (inside `logs/` folder).
        - verbose (bool): Whether to print logs to console.
        - maxBytes (int): Maximum log file size before rotating.
        - backupCount (int): Number of old log files to keep.
        """
        self.verbose = verbose
        log_path = os.path.join("logs", log_filename)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if log_path in Logger.loggers:
            self.logger = Logger.loggers[log_path]
            return
        
        # Set up logger
        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)
        
        # Create rotating file handler
        handler = RotatingFileHandler(log_path, maxBytes=maxBytes, backupCount=backupCount)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # Store the logger instance
        Logger.loggers[log_path] = self.logger

    def log(self, message, level="INFO"):
        """Logs messages with specified level."""
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        if self.verbose:
            print(message)

    def error(self, message):
        """Logs error messages."""
        self.log(message, level="ERROR")
