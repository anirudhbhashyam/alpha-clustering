import os
import sys
import time
import shutil
import argparse
from pathlib import Path

from functools import partial

import multiprocessing

from typing import Iterator

import numpy as np
import pandas as pd

import scipy

from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.pyplot.switch_backend("Agg")

from alpha_clustering.alpha_complex import AlphaComplexND
from alpha_clustering.cluster import Cluster
from alpha_clustering.plot import Plot
from alpha_clustering.io_handler import IOHandler

CPD = Path(__file__).resolve().parents[1]

sys.path.append(str(CPD))

from utils import convert_clusters_format_to_sklearn 

ONE_MILLION = 1000_000

TERMINAL_WIDTH, _ = shutil.get_terminal_size()


def read_glacier_data(file: Path, chunksize: int | None) -> Iterator[pd.DataFrame]:
    data_iter = pd.read_csv(file, sep = "\t", header = None, chunksize = chunksize)
    yield from data_iter


def process_data_chunk(data_chunk: pd.DataFrame) -> np.ndarray:
    data_df = data_chunk.iloc[:, : 3]
    point_cloud = data_df.to_numpy()
    yield point_cloud


def create_alpha_complex(
    point_cloud: np.ndarray, 
    alpha: float,
) -> list[np.array]:
    ac = AlphaComplexND(point_cloud)
    ac.fit()
    ac.predict(alpha)
    # triangles = ac.get_complex[0]
    return ac.get_complex


def cluster_cloud(shape: list[np.ndarray]) -> list[list[int]]:
    clustering = Cluster(shape)
    clustering.fit()
    return clustering.predict()
    

def plot_simplices(point_cloud: np.ndarray, simplices: np.ndarray, ax: plt.Axes) -> None:
    ax.plot_trisurf(
        *point_cloud.T,
        triangles = simplices, 
        linewidth = 1.0, 
        color = sns.color_palette("mako", 50)[20]
    )

def run_on_file_full(file: Path) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list[int]], list[list[np.array]]
    ]:
    data = pd.read_csv(file, sep = "\t", header = None)
    cloud = next(process_data_chunk(data))
    print(f"Processed point cloud data.".center(TERMINAL_WIDTH, "-"))

    start_time = time.time()
    shape = create_alpha_complex(cloud, 5e-6)
    clusters = cluster_cloud(shape)
    end_time = time.time()
    print(f"Finished alpha clustering.".center(TERMINAL_WIDTH, "-"))


    # plot = Plot(cloud)
    # fig_cluster = plot.clusters(clusters)
    # fig_shape = plot.alpha_complex(shape)
    cluster_details = pd.DataFrame(
        {
            "$n$-clusters": len(clusters),
            "$n$-points": len(cloud),
            "$n$-simplices": sum(len(s) for s in shape),
            "AC Time": abs(start_time - end_time)
        },
        index = [0]
    )
    # print(f"Finished plotting.".center(TERMINAL_WIDTH // 2, "-"))

    start_time = time.time()
    kmeans = KMeans(n_clusters = len(clusters))
    kmeans_clusters = kmeans.fit_predict(cloud)
    end_time = time.time()
    print(f"Finished kmeans clustering.".center(TERMINAL_WIDTH // 2, "-"))

    kmeans_cluster_details = pd.DataFrame(
        {
            "$n$-clusters": len(np.unique(kmeans_clusters)),
            "KMeans Time": abs(start_time - end_time)
        },
        index = [0]
    )

    return cloud, cluster_details, kmeans_cluster_details, clusters, shape


# def run_on_file_chunks(file: Path, cs: int) -> tuple[pd.DataFrame, plt.Figure]:
#     partial_alpha_shape_creator = partial(create_alpha_shape, alpha = 5e-6)
#     clouds = [process_data_chunk(c) for c in read_glacier_data(file, cs)]
#     # print("Processed point clouds.".center(TERMINAL_WIDTH, "-"))
#     kmeans = KMeans(n_clusters = len(clusters))
#     shapes = dict()
# 
#     start_time = time.perf_counter()
#     for i, data_chunk in enumerate(clouds):
#         try:
#             # cloud = process_data_chunk(data_chunk)
#             shapes[i] = partial_alpha_shape_creator(cloud)
#         except scipy.spatial._qhull.QhullError as e:
#             continue
# 
#     cumulative_shape = list()
#     for i in range(len(shapes[0])):
#         cumulative_shape.append(np.concatenate([s[i] for s in shapes.values()]))
# 
#     clusters = cluster_cloud(cumulative_shape)
#     end_time = time.perf_counter()
# 
#     # for ind, shape in shapes.items():
#     #     plot_simplices(clouds[ind], shape[0], ax)
#     
# 
#     cumulative_cloud = np.concatenate(clouds)
#     plot = Plot(cumulative_cloud)
#     fig = plot.clusters(clusters)
# 
#     cluster_details = pd.DataFrame(
#         {
#             "$n$-clusters": len(clusters),
#             "AC Time": start_time - end_time
#         }, 
#         index = [0]
#     )
# 
#     start_time = time.perf_counter()
#     kmeans_clusters = list()
#     for i, cloud in enumerate(clouds):
#         clusters = kmeans.fit_predict(cloud)
#     end_time = time.perf_counter()
# 
#     kmeans_cluster_details = pd.DataFrame(
#         {
#             "$n$-clusters": len(kmeans_clusters),
#             "$n$-points": len(cloud),
#             "$n$-simplices": sum(len(s) for s in shape),
#             "KMeans Time": start_time - end_time
#         },
#         index = [0]
#     )
# 
#     return cluster_details, fig
# 

def process_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument(
        "data_dir",
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
    io_h = IOHandler(
        data_dir = Path(args.data_dir).resolve(),
        save_dir = CPD / "results" / "glacier",
    )
    if args.chunksize is None:
        chunksize = ONE_MILLION
    else:
        chunksize = int(args.chunksize)

    dataset = "Hochebenkar_TLS_20190624"

    cluster_details_dfs = list()
    kmeans_cluster_details_dfs = list()
    figs = list()
    figs_shapes = list()

    # with multiprocessing.Pool(4) as pool:
    #     cluster_details, kmeans_cluster_details, fig = pool.map(
    #         run_on_file_full, io_h.get_data_dir.glob(f"*.xyz")
    #     )
    #     cluster_details_dfs.append(cluster_details)
    #     kmeans_cluster_details_dfs.append(kmeans_cluster_details)
    #     figs.append(fig)

    fig, ax = plt.subplots(1, 1, figsize = (16, 9), dpi = 300, subplot_kw = {"projection": "3d"})
    cm = sns.color_palette("mako", 50)[20]

    for file in io_h.get_data_dir.glob("*.xyz"):
        if "4a" in file.stem:
            continue
        print(f"Processing file: {file.name}".center(TERMINAL_WIDTH, "="))
        cloud, cluster_details, kmeans_cluster_details, alpha_clusters, alpha_complex = run_on_file_full(file)
        cluster_details_dfs.append(cluster_details)
        kmeans_cluster_details_dfs.append(kmeans_cluster_details)
        triangles = [s for s in alpha_complex if s.shape[1] == 3]
        ax.plot_trisurf(
            *cloud.T,
            triangles = triangles,
            label = file.stem,
            alpha = 0.8,
            linewidth = 0.0,
            color = cm
        )
        ax.grid(False)

        fig.savefig(
            CPD / "results" / "glacier" / "figs" / f"{dataset}_shape_timelapse_{file.stem}.png",
            dpi = 300,
            bbox_inches = "tight"
        )

        # fig_cluster.savefig(
        #     CPD / "results" / "glacier" / "figs" / f"{dataset}_cluster_{file.stem}.png",
        #     dpi = 300,
        #     bbox_inches = "tight"
        # )
        # fig_shape.savefig(
        #     CPD / "results" / "glacier" / "figs" / f"{dataset}_shape_{file.stem}.png",
        #     dpi = 300,
        #     bbox_inches = "tight"
        # )
        print(f"Processed file: {file.name}".center(TERMINAL_WIDTH, "="))
        break

    io_h.write_results(
        dataset,
        cluster_details_dfs,
        f"{dataset}_evaluation.tex",
        "Glacier clustering evaluation.",
        join_axis = 0
    )

    io_h.write_results(
        dataset,
        kmeans_cluster_details_dfs,
        f"{dataset}_kmeans_evaluation.tex",
        "Glacier kmeans clustering evaluation.",
        join_axis = 0
    )

    # io_h.write_figs(
    #     dataset,
    #     figs,
    #     [f"{dataset}_clusters_{i}" for i in range(len(figs))],
    # )

    # io_h.write_figs(
    #     dataset,
    #     figs_shapes,
    #     [f"{dataset}_shape_{i}" for i in range(len(figs_shapes))],
    # )

    # cluster_details, kmeans_cluster_details, fig = run_on_file_chunks(file, chunksize)

    # with multiprocessing.Pool(os.cpu_count()) as p:
    #     try:
    #         i, shape = p.map(partial_alpha_shape_creator, zip(range(len(clouds)), clouds))
    #         shapes[i] = shape
    #     except scipy.spatial._qhull.QhullError as e:
    #         pass
    # ax.view_init(elev=10., azim=ii)
    # for i, chunk in enumerate(read_glacier_data(file, chunksize)):
    #     print(f"Processing chunk {i + 1}...")
    #     if i > 0:
    #         break
        
    #     try:
    #         create_alpha_shape_chunk(chunk, 5e-10, ax)
    #     except scipy.spatial._qhull.QhullError as e:
    #         continue

    fig.savefig("glacier.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
