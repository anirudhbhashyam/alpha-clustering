import shutil
import argparse
from pathlib import Path

import multiprocessing

from typing import Iterator

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from alpha_clustering.alpha_shape import AlphaShapeND
from alpha_clustering.cluster import Cluster

ONE_MILLION = 1000000

TERMINAL_WIDTH, _ = shutil.get_terminal_size()


def read_glacier_data(file: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    data_iter = pd.read_csv(file, sep = "\t", header = None, chunksize = chunksize)
    yield from data_iter


def process_data_chunk(data_chunk: pd.DataFrame) -> np.ndarray:
    print(data_chunk.iloc[0])
    print(data_chunk.columns)
    data_df = data_chunk.iloc[:, 3 :]
    print(data_df.iloc[0])
    point_cloud = data_df.to_numpy()
    return point_cloud

def create_alpha_shape_chunk(file: Path, alpha: float, data_chunk: pd.DataFrame) -> None:
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111, projection = "3d")
    point_cloud = process_data_chunk(data_chunk)
    ac = AlphaShapeND(point_cloud)
    ac.fit()
    ac.predict(5e-6)
    triangles = ac.get_shape[0]
    ax.plot_trisurf(
        *point_cloud.T,
        triangles = triangles, 
        linewidth = 1.0, 
        color = sns.color_palette("mako", 50)[20]
    )
        # clustering = Cluster(ac.get_shape)
        # clustering.fit()
        # clusters = 
    fig.savefig("glacier.png", bbox_inches = "tight")



def process_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument(
        "data_file",
        type = str
    )
    args.add_argument(
        "--chunksize",
        "-cs",
        type = str,
        default = None,
        required = False
    )
    return args.parse_args()


def main() -> int:
    args = process_args()
    file = args.data_file
    if args.chunksize is None:
        chunksize = ONE_MILLION
    else:
        chunksize = int(args.chunksize)
    # Process three chunks.
    # with multiprocessing.Pool(4) as p:
    #     p.starmap(create_alpha_shape_chunk, zip([file] * 3, [5e-6] * 3, read_glacier_data(file, chunksize)))
    first_chunk = next(read_glacier_data(file, chunksize))
    first_point_cloud = process_data_chunk(first_chunk)
    
    return 0


if __name__ == "__main__":
    SystemExit(main())
