from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from scipy.io import arff

import matplotlib.pyplot as plt

from .logger import Logger

from .exceptions import UnsupportedFileType

LOGGER = Logger(__name__)

@dataclass
class IOHandler:
    data_dir: Path
    save_dir: Path

    def __post_init__(self) -> None:
        if not self.save_dir.exists():
            self.save_dir.mkdir(exist_ok = True)

    def load_point_cloud(self, filename: str) -> pd.DataFrame:
        LOGGER.info(f"Loading point cloud '{filename}' from '{self.data_dir}'.")
        to_load = self.data_dir.joinpath(filename)
        if to_load.suffix == ".arff":
            dataset = arff.loadarff(to_load)
            df = pd.DataFrame(dataset[0])
        elif to_load.suffix == ".csv":
            df = pd.read_csv(to_load)
        else:
            raise UnsupportedFileType(to_load.suffix, LOGGER)
        return df

    def write_figs(
        self, 
        dataset: str,
        figs: list[plt.Figure], 
        save_names: list[str]
    ) -> None:
        if len(figs) != len(save_names):
            raise ValueError("The number of figures and save names must be equal.")
        write_dir = self._create_dir(Path("figs") / f"{dataset}-figures")
        for fig, save_name in zip(figs, save_names):
            LOGGER.info(f"Writing figure '{save_name}' to '{write_dir}'.")
            fig.savefig(
                write_dir / save_name,
                dpi = 400,
                bbox_inches = "tight"
            )

    def write_results(
        self, 
        dataset: str, 
        results: list[pd.DataFrame],
        save_name: str,
        caption: str,
        join_axis: int = 1,
        position: str = "!htbp",
        column_format: str = None,
        label = None
    ) -> None:
        write_dir = self._create_dir(Path("metrics") / f"{dataset}-evaluation")
        caption_style = {
            "selector": "caption",
            "props": [
                ("caption-side", "bottom"),
            ]
        }
        main_df = pd.concat(results, axis = join_axis)
        LOGGER.info(f"Writing result '{save_name}' to '{write_dir}'.")
        with open(write_dir / save_name, "w") as f:
            f.write(
                main_df.style\
                # .apply(lambda x: [f"{v:.2f}" for v in x], subset = format_subset)\
                .applymap_index(
                    lambda v: "font-weight: bold;", axis = "columns"
                )\
                .set_caption(caption)\
                .set_table_styles([caption_style])\
                .to_latex(
                    column_format = column_format,
                    position = position,
                    position_float = "centering",
                    hrules = True,
                    caption = caption,
                    convert_css = True,
                    label = label
                )
            ) 
    
    @property
    def get_data_dir(self) -> Path:
        return self.data_dir
    
    def _create_dir(self, name: str | Path) -> Path:
        dir_to_create = Path(self.save_dir, name)
        if not dir_to_create.exists():
            dir_to_create.mkdir(exist_ok = True, parents = True)
        return dir_to_create
