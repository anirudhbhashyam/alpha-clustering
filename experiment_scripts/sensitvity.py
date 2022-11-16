import os
import argparse
from pathlib import Path

from typing import Iterable

import numpy as np
import pandas as pd

from alpha_clustering.alpha_shape import AlphaShape, AlphaShapeND
from alpha_clustering.cluster import Cluster
from alpha_clustering.io_handler import IOHandler

CPD = Path(__file__).resolve().parents[1]
        

def analyse_sensitivity_fix_data(points: np.ndarray, alphas: Iterable[float]) -> list[int]:
    ac = AlphaShapeND(points)
    ac.fit()
    n_predicted_clusters = list()
    for alpha in alphas:
        ac.predict(alpha)
        clustering = Cluster(ac.get_shape)
        clustering.fit()
        n_clusters = len(clustering.predict_iter(4))
        n_predicted_clusters.append(n_clusters)
    return n_predicted_clusters


def analyse_sensitivity_fix_alpha(points: list[np.ndarray], alpha: float) -> list[int]:
    n_predicted_clusters = list()
    for point_cloud in points:
        ac = AlphaShapeND(point_cloud)
        ac.fit()
        ac.predict(alpha)
        clustering = Cluster(ac.get_shape)
        clustering.fit()
        n_clusters = len(clustering.predict(4))
        n_predicted_clusters.append(n_clusters)
    return n_predicted_clusters



def get_sensitivity_results(alphas: list[float], n_predicted_clusters: list[int]) -> pd.DataFrame:
    df = pd.DataFrame()
    pass


def process_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_name",
        "-dn",
        type = "str",
        required = False
    )
    return args.parse_args()
    

def main() -> int:
    # args = process_args()
    # dataset_name = args.dataset_name
    io_handler = IOHandler(
        CPD / "data" / "points",
        Path(CPD / "results" / "sensitivity")
    )
    # data_df = io_handler.load_point_cloud(dataset_name)
    density_test_clouds = list()
    for i in range(1, 11):
        data_df = io_handler.load_point_cloud(f"sphere_dense_pts_{i}.csv")
        density_test_clouds.append(data_df.iloc[:, : -1].to_numpy())
        
    # points = data_df.iloc[:, : -1].to_numpy()

    alphas = np.arange(0.0, 5.0, 0.01)
    n_clusters_predicted = analyse_sensitivity_fix_alpha(np.array(density_test_clouds), alpha = 1.00)
    print(n_clusters_predicted)

    return 0

if __name__ == "__main__":
    SystemExit(main())