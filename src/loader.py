from pathlib import Path

import pandas as pd

from scipy.io import arff

class UnsupportedFileType(ValueError):
    def __init__(self, file_type):
        self.file_type = file_type
        self.message = f"Unsupported data file type: {file_type}"
        super().__init__(self.message)

def load_point_cloud(data_path: Path) -> pd.DataFrame:
    if data_path.suffix == ".arff":
        dataset = arff.loadarff(data_path)
        df = pd.DataFrame(dataset[0])
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise UnsupportedFileType(data_path.suffix)
    return df