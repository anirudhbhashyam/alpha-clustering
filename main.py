#!/usr/bin/env python
import sys
import argparse
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CPD = Path(__file__).resolve().parent

sys.path.append(CPD.joinpath("src").as_posix())

from alpha_shape import AlphaShape, AlphaShape2D, AlphaShape3D
from cluster import Cluster, ClusterEvaluate
from config import Config
from io_handler import IOHandler
from plot import Plot

def convert_clusters_format_to_sklearn(n_points: int, clusters: list[list[int]]) -> np.array:
    new_clusters = np.zeros(n_points)
    for i, cluster in enumerate(clusters):
        new_clusters[list(cluster)] = i
    return np.sort(new_clusters)

def convert_sklearn_format_to_clusters(clusters: np.array) -> list[list[int]]:
    new_clusters = list()
    for i in np.unique(clusters):
        new_clusters.append(np.where(clusters == i)[0])
    return new_clusters

def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int]:
    points = df.iloc[:, : -1].to_numpy()
    all_labels = df.iloc[:, -1].apply(lambda x: int(x))
    n_true_clusters = len(all_labels.unique())
    return points, all_labels, n_true_clusters

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Cluster data using alpha shape clustering."
    )
    parser.add_argument(
        "dataset",
        help = "Dataset file path (including extension).",
        type = str
    )
    parser.add_argument(
        "alpha", 
        help = "Alpha Parameter", 
        type = float
    )
    return parser.parse_args()

def find_clusters(data: pd.DataFrame, alpha: float) \
    -> tuple[np.ndarray, AlphaShape, Cluster, sklearn.cluster]:

    
    points, _, n_true_clusters = process_data(data)
    points_dimension = points.shape[1]

    if points_dimension == 2:
        ac = AlphaShape2D(points, alpha)
    elif points_dimension == 3:
        ac = AlphaShape3D(points, alpha)
    else:
        raise ValueError("Points dimension must be 2 or 3.")
    ac.fit()

    kmeans = KMeans(
        n_clusters = np.random.default_rng().integers(2, 5 + 1, 1)[0],
        random_state = 15485863
    )
    clustering = Cluster(
        shape = ac.get_shape
    )
    clustering.fit()

    return points, ac, clustering, kmeans 

def create_plots(
    dataset: str,
    io: IOHandler,
    points: np.ndarray,
    alpha_shape: np.ndarray,
    predicted_clusters: list[list[int]],
    other_clusters: np.array
) -> None:

    plots = Plot(vertices = points)

    fig0 = plots.points_scatter()

    fig1 = plots.alpha_shape(
        alpha_shape,
        (16, 9),
        points_q = True, 
        ticks_q = True
    )

    fig2 = plots.clusters(
        predicted_clusters,
        (16, 9),
        ticks_q = True
    )

    fig3 = plots.clusters(
        other_clusters,
        (16, 9),
        ticks_q = True
    )

    io.write_figs(
        dataset,
        [fig0, fig1, fig2, fig3],
        ["scatter.png", "alpha_shape.png", "alpha_shape_clusters.png", "kmeans_clusters.png"]
    )    

def evaluate_clusters(
    dataset: str,
    io: IOHandler,
    points: np.ndarray,
    true_labels: np.array,
    predicted_labels: np.array, 
    other_clustering: tuple[str, np.array],
) -> None:

    method, other_labels = other_clustering

    cl_eval_one = ClusterEvaluate(
        points,
        true_labels,
        predicted_labels
    )

    cl_eval_two = ClusterEvaluate(
        points,
        true_labels,
        other_labels
    )
    
    df1 = cl_eval_one.get_results(method = "alpha shape clustering")
    df2 = cl_eval_two.get_results(method = f"{method} clustering")

    io.write_results(
        dataset,
        [df1, df2],
        f"{dataset}_evaluation.tex",
        caption = "Summary of the evaluation of the clustering methods."
    )

def summarise_points(
    dataset: str, 
    io: IOHandler, 
    alpha_obj: AlphaShape,
    other_clustering_obj: sklearn.cluster,
    clustering_obj: Cluster
) -> None:
    df1 = alpha_obj.get_summary(dataset)
    df2 = clustering_obj.get_summary(
        dataset, 
        other_cluster_data = [
            (f"Number of {other_clustering_obj.__class__.__name__} clusters", len(np.unique(other_clustering_obj.labels_)))
        ],
        threshold = 4
    )
    io.write_results(
        dataset,
        [df1, df2],
        f"{dataset}_summary.tex",
        caption = "Summary of the dataset and the alpha shape.",
        join_axis = 0
    )

def main() -> None:
    args = process_args()
    config = Config(
        config_dir = CPD / "config",
        filenames = ["settings.json"]
    )

    config_data = next(config.load())

    io = IOHandler(
        data_dir = Path(config_data["PATH"]["DATA_POINTS_PATH"]).resolve(),
        save_dir = CPD / "results"
    )

    data = io.load_point_cloud(args.dataset)
    dataset_name = args.dataset.split(".")[0]
    _, true_labels, n_true_clusters = process_data(data)

    points, ac_obj, clustering, other_clustering = find_clusters(data, args.alpha)

    alpha_shape = ac_obj.get_shape
    predicted_clusters = clustering.predict(10)
    other_clusters = other_clustering.fit_predict(points)
    # print(alpha_shape)

    create_plots(
        dataset_name, 
        io, 
        points, 
        alpha_shape,
        predicted_clusters, 
        convert_sklearn_format_to_clusters(other_clusters)
    )

    evaluate_clusters(
        dataset_name, 
        io, 
        points, 
        true_labels, 
        convert_clusters_format_to_sklearn(len(points), predicted_clusters), 
        ("kmeans", other_clusters)
    )

    summarise_points(
        dataset_name,
        io,
        ac_obj,
        other_clustering,
        clustering
    )

if __name__ == "__main__":
    main() 
