import os 
import json
from pathlib import Path

from dataclasses import dataclass

@dataclass
class Config:
    """
    General configuration parser class.
    """
    config_dir: Path
    filenames: list[str]

    def __read_file(self, filename: str):
        """
        Read a single configuration file and yield a dictionary.
        """
        with self.config_dir.joinpath(filename).open("r") as f:
           yield json.load(f)
    
    def load(self) -> dict[str, str]:
        """
        Read all files and yield dictionaries.
        """
        for file in self.filenames:
            yield from self.__read_file(file)

    