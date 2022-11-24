import time

from functools import wraps

from pathlib import Path

from typing import Iterable, Any, Callable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from scipy.stats import ttest_ind

from alpha_clustering.alpha_shape import AlphaShapeND
from alpha_clustering.cluster import Cluster
from alpha_clustering.io_handler import IOHandler

CPD = Path(__file__).resolve().parents[1]

ALL_DATASETS: dict[str, tuple[float, int]] = {
    "aggregation.arff": (1.625, 7),
    "2d-20c-no0.arff": (1.835, 20),
    "st900-2-9.arff": (6.327, 9),
    "complex9.arff": (0.124, 9),
    "spectral.arff": (0.486, 7),
    "test.arff": (0.499, 9),
    "test_2.arff": (0.105, 8),
    "test_3.arff": (187.719, 6),
    "bio-protein.arff": (0.005, 5),
    "hypercube.arff": (4.37575, 8),
    "chainlink.arff": (2.500, 2),
    "golf-ball.arff": (0.624, 1),
    "tesseract.arff": (4.25, 2),
    "hepta.arff": (1.249, 7),
}


class UnequalVarianceError(Exception):
    pass


# def timing(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
#     @wraps(f)
#     def wrap(*args, **kwargs) -> float:
#         start = time.perf_counter()
#         result = f(*args, **kwargs)
#         return time.perf_counter() - start
#     return wrap

def time_experiment(
    points: np.ndarray,
    alpha: float,
    k_means: int,
    runs: int
) -> tuple[float, float, int, int]:
    alpha_time = 0.0
    kmeans_time = 0.0
    ac = AlphaShapeND(points)
    for _ in range(runs):
        start = time.perf_counter()
        ac.fit()
        ac.predict(alpha)
        clustering = Cluster(
            shape = ac.get_shape
        )
        clustering.fit()
        clusters_alpha = clustering.predict(4)
        alpha_time += time.perf_counter() - start

        start = time.perf_counter() 
        kmeans = KMeans(k_means)
        kmeans.fit_predict(points)
        kmeans_time += time.perf_counter() - start

    # Store the number of clusters found by the algorithm.

    return alpha_time / runs, kmeans_time / runs, len(clusters_alpha), kmeans.n_clusters

def get_significance(series_1: Iterable[float], series_2: Iterable[float]) -> tuple[float, float]:
    # Perform a T-test to determine if the difference between the two scores is significant.
    
    # Calculate the standard deviation of the scores.
    var_series_1 = np.std(series_1) ** 2
    var_series_2 = np.std(series_2) ** 2

    if var_series_1 > var_series_2:
        f_stat = var_series_1 / var_series_2
    else:
        f_stat = var_series_2 / var_series_1

    # if f_stat >= 4.0:
    #     raise UnequalVarianceError(f"The variances of the two score sets, {var_series_1} and {var_series_2} are not close enough. The F-stat is {f_stat}.")
    
    # Perform the T-test.
    t_stat, p_value = ttest_ind(series_1, series_2, equal_var = True)

    return t_stat, p_value

def main() -> int:
    io_handler = IOHandler(
        CPD / "data" / "points",
        Path(CPD / "results" / "performance")
    )

    dfs = []

    for dataset_name, vals in ALL_DATASETS.items():
        print(f"{dataset_name}".center(80, "="))
        alpha, k_means = vals
        data = io_handler.load_point_cloud(dataset_name)
        points = data.iloc[:, : -1].to_numpy()
        if dataset_name == "tesseract.arff":
            points = data.iloc[:, : -2].to_numpy()
        alpha_time, kmeans_time, n_clusters_alpha, n_clusters_kmeans = time_experiment(points, alpha, k_means, 10)
        dataset_name = dataset_name.replace(".arff", "")
        dataset_name = dataset_name.replace("_", "-")
        df = pd.DataFrame(
            {
                "alpha clustering time": alpha_time,
                "kmeans clustering time": kmeans_time,
                "predicted alpha clusters": n_clusters_alpha,
                "predicted kmeans clusters": n_clusters_kmeans,
                "ground truth clusters": data.iloc[:, -1].nunique()
            },
            index = [dataset_name],
        )
        df.astype({
            "predicted alpha clusters": int,
            "predicted kmeans clusters": int,
            "ground truth clusters": int
        }) 
        dfs.append(df)

    df = pd.concat(dfs)

    time_results = df.iloc[:, : 2]
    cluster_results = df.iloc[:, 2 :]

    io_handler.write_results(
        dataset = "all",
        results = [time_results],
        save_name = "performance_results.tex",
        caption = "Performance results for all datasets.",
        join_axis = 0,
    )

    io_handler.write_results(
        dataset = "all",
        results = [cluster_results],
        save_name = "cluster_results.tex",
        caption = "Cluster results for all datasets.",
        join_axis = 0,
    )

    t_stat, p_value = get_significance(
        df["alpha clustering time"],
        df["kmeans clustering time"]
    )
    print(f"t-stat: {t_stat}, p-value: {p_value}")

    return 0 

if __name__ == "__main__":
    SystemExit(main())