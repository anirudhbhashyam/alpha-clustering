import sys
import argparse
from pathlib import Path

from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from alpha_clustering.alpha_complex import AlphaComplexND
from alpha_clustering.cluster import Cluster
from alpha_clustering.plot import Plot
from alpha_clustering.io_handler import IOHandler

CPD = Path(__file__).resolve().parents[1]

sys.path.append(str(CPD))

from utils import *
        

def analyse_sensitivity_fix_data(
    points: np.ndarray, 
    alphas: Iterable[float],
    true_labels: np.ndarray = None
) -> tuple[list[int], list[int], list[float]]:
    ac = AlphaComplexND(points)
    ac.fit()
    n_predicted_clusters = list()
    n_simplices = list()
    mi_scores = list()
    n_points = points.shape[0]
    for alpha in alphas:
        ac.predict(alpha)
        clustering = Cluster(ac.get_complex)
        clustering.fit()
        clusters = clustering.predict(4)
        n_clusters = len(clusters)
        
        n_simplices.append(ac.n_simplices)
        n_predicted_clusters.append(n_clusters)
        mi_scores.append(
            metrics.mutual_info_score(true_labels, convert_clusters_format_to_sklearn(n_points, clusters))
        )
    return n_predicted_clusters, n_simplices, mi_scores


def analyse_sensitivity_fix_alpha(
    points: list[np.ndarray], 
    alpha: float
) -> tuple[list[int], list[int], list[int], list[plt.Figure]]:
    n_points = list()
    n_predicted_clusters = list()
    n_simplices = list()
    alpha_figs = list()
    for point_cloud in points:
        plot = Plot(vertices = point_cloud)
        ac = AlphaComplexND(point_cloud)
        ac.fit()
        ac.predict(alpha)
        clustering = Cluster(ac.get_complex)
        clustering.fit()
        n_clusters = len(clustering.predict(4))

        n_predicted_clusters.append(n_clusters)
        n_simplices.append(ac.n_simplices)
        n_points.append(point_cloud.shape[0])
        alpha_figs.append(plot.alpha_complex(ac.get_complex, points_q = False))

    return n_predicted_clusters, n_simplices, n_points, alpha_figs


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
    dataset_varying_density = "random_spheres"
    dataset_varying_alpha = "hepta"

    density_test_clouds = list()
    varying_alpha_data_df = io_handler.load_point_cloud(f"{dataset_varying_alpha}.arff")
    varying_alpha_points = varying_alpha_data_df.iloc[:, : -1].to_numpy()
    varying_alpha_true_labels = varying_alpha_data_df.iloc[:, -1].apply(lambda x: int(x)).to_numpy()
    for i in range(1, 21):
        data_df = io_handler.load_point_cloud(f"sphere_dense_pts_{i}.csv")
        density_test_clouds.append(data_df.iloc[:, : -1].to_numpy())
        
    alphas = np.arange(0.1, 5.5, 0.5)
    n_clusters_predicted, n_simplices, n_points, alpha_figs = analyse_sensitivity_fix_alpha(np.array(density_test_clouds), alpha = 1.00)
    n_clusters_predicted_varying_alpha, n_simplices_varying_alpha, mi_scores_varying_alpha = analyse_sensitivity_fix_data(varying_alpha_points, alphas, varying_alpha_true_labels)

    varying_density_df = pd.DataFrame(
        {
            "# points": [int(x) for x in n_points],
            "Densities": [1.26645, 3.49977, 4.66239, 6.22989, 8.32099, 10.3166, 10.8834, 12.1598, 14.8071, 16.0678, 16.9286, 18.2762, 19.3027, 23.7176, 23.2686, 24.9328, 26.4615, 28.8363, 29.2021, 30.6723],
            "# simplices": [int(x) for x in n_simplices],
            "# clusters predicted": n_clusters_predicted
        }
    )

    varying_alpha_df = pd.DataFrame(
        {
            "Alpha": alphas,
            "# simplices": n_simplices_varying_alpha,
            "# clusters predicted": n_clusters_predicted_varying_alpha,
            "MI scores": mi_scores_varying_alpha
        }
    )

    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        varying_density_df["# points"].to_numpy().reshape(-1, 1),
        varying_density_df["# simplices"].to_numpy(),
        test_size = 0.2,
        random_state = 42
    )
    lr.fit(
        X_train, 
        y_train
    )

    lr_df = pd.DataFrame(
        {
            "coef": lr.coef_,
            "intercept": lr.intercept_,
            "r2": lr.score(X_test, y_test)
        }
    )

    change_style_format_df(varying_alpha_df, lambda x: f"{x:.2f}", col = "Alpha")
    change_style_format_df(varying_alpha_df, lambda x: f"{x:.2f}", col = "MI scores")
    change_style_format_df(varying_alpha_df, lambda x: f"{x:.0f}", col = "# simplices")
    change_style_format_df(varying_alpha_df, lambda x: f"{x:.0f}", col = "# clusters predicted")

    change_style_format_df(varying_density_df, lambda x: f"{x:.2f}", col = "Densities")
    change_style_format_df(varying_density_df, lambda x: f"{x:.0f}", col = "# points")
    change_style_format_df(varying_density_df, lambda x: f"{x:.0f}", col = "# simplices")
    change_style_format_df(varying_density_df, lambda x: f"{x:.0f}", col = "# clusters predicted")

    change_style_format_df(lr_df, lambda x: f"{x:.2f}", col = "coef")
    change_style_format_df(lr_df, lambda x: f"{x:.2f}", col = "intercept")
    change_style_format_df(lr_df, lambda x: f"{x:.2f}", col = "r2")

    io_handler.write_figs(
        dataset_varying_density,
        alpha_figs,
        [f"alpha_complex_{i}.png" for i in range(len(alpha_figs))],
    )

    io_handler.write_results(
        dataset_varying_density,
        [varying_density_df],
        f"{dataset_varying_density}_evaluation.tex",
        caption = "Evaluation of the alpha complex on a varying density of points.",
        label = "TAB:VaryingDensityResults",
        join_axis = 0
    )

    io_handler.write_results(
        dataset_varying_density,
        [lr_df],
        f"{dataset_varying_density}_linear_regression.tex",
        caption = "Linear regression of the number of simplices and the density of points.",
        label = "TAB:VaryingDensityRegressionResults",
        join_axis = 0
    )

    io_handler.write_results(
        dataset_varying_alpha,
        [varying_alpha_df],
        f"{dataset_varying_alpha}_evaluation.tex",
        caption = "Evaluation of the alpha complex on a varying alpha.",
        label = "TAB:VaryingAlphaResults",
        join_axis = 0
    )
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())