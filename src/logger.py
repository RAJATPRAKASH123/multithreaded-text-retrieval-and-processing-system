import os
from datetime import datetime

class Logger:
    """Handles logging for errors and events using RotatingFileHandler."""
    def __init__(self, log_file="logs/pipeline.log", verbose=False, maxBytes=5*1024*1024, backupCount=5):
        self.verbose = verbose
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        import logging
        from logging.handlers import RotatingFileHandler
        self.logger = logging.getLogger("PipelineLogger")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(log_file, maxBytes=maxBytes, backupCount=backupCount)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        # To prevent duplicate handlers, remove any existing ones.
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log(self, message, level="INFO"):
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        if self.verbose:
            print(message)
    
    def error(self, message):
        self.log(message, level="ERROR")
