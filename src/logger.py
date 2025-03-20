import os
from datetime import datetime

class Logger:
    """Handles logging for errors and events."""
    def __init__(self, log_file="logs/pipeline.log", verbose=True):
        self.log_file = log_file
        self.verbose = verbose
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message, level="INFO"):
        """Logs messages with timestamps and levels (INFO, ERROR)."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level} - {message}"
        
        if self.verbose:
            print(log_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def error(self, message):
        """Logs error messages."""
        self.log(message, level="ERROR")
