import logging 
from pathlib import Path

class Logger:
    def __init__(self, name: str):
        self.name = name
        self.log_dir = Path("logs").resolve()
        self.log_file_path = Path(self.log_dir, f"{self.name}.log")
        if not self.log_dir.exists():
            self.log_dir.mkdir(exist_ok = True)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            "%(levelname)s \t %(asctime)s %(message)s", 
            "%d-%m-%y %H:%M:%S"
        )
        self.console_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(self.log_file_path)
        
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        
        self.logger.debug(f"Logger for {self.name} created.")
        
    def debug(self, message: str):
        self.logger.debug(message)
        
    def info(self, message: str):
        self.logger.info(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
        
    def critical(self, message: str):
        self.logger.critical(message)
        
    def exception(self, message: str):
        self.logger.exception(message)
        
    def log(self, message: str):
        self.logger.log(message)
        
    def set_level(self, level: int):
        self.logger.setLevel(level)
        
    def get_level(self) -> int:
        return self.logger.getEffectiveLevel()
        
    def set_file(self, filename: str):
        self.logger.addHandler(logging.FileHandler(filename))
        
    def remove_file(self):
        self.logger.removeHandler(self.logger.handlers[1])

    def add_null_handler(self):
        self.logger.addHandler(logging.NullHandler())
        
    def remove_all_handlers(self):
        self.logger.handlers.clear()
        
    @property
    def get_name(self) -> str:
        return self.name