import time

from functools import wraps

from pathlib import Path

from typing import Iterable, Any, Callable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from scipy.stats import ttest_ind

from alpha_clustering.alpha_shape import AlphaShape2D, AlphaShape3D
from alpha_clustering.cluster import Cluster, ClusterEvaluate
from alpha_clustering.io_handler import IOHandler
from alpha_clustering.plot import Plot
from alpha_clustering.config import Config

CPD = Path(__file__).resolve().parent

ALL_DATASETS: dict[str, tuple[float, int]] = {
    "aggregation.arff": (-0.6500, 7),
    "2d-20c-no0.arff": (-1.5678, 20),
    "st900-2-9.arff": (-12.5630, 9),
    "complex9.arff": (-0.0033, 9),
    "spectral.arff": (-0.06124, 7),
    "test.arff": (-0.0156, 9),
    "test_2.arff": (-0.0030, 8),
    "test_3.arff": (-9400.0000, 6),
    "bio-protein.arff": (0.0053, 5),
    "hypercube.arff": (5.0000, 8),
    "chainlink.arff": (5.0000, 2),
    "golf-ball.arff": (0.1000, 1),
    "tesseract.arff": (2.0000, 2),
    "hepta.arff": (2.0000, 7),
}


class UnequalVarianceError(Exception):
    pass


def timing(f: Callable[[Any, ...], Any]) -> Callable[[Any, ...], Any]:
    @wraps(f)
    def wrap(*args, **kwargs) -> float:
        start = time.perf_counter()
        result = f(*args, **kwargs)
        return time.perf_counter() - start
    return wrap

def time_experiment(
    points: np.ndarray,
    alpha: float,
    k_means: int,
    runs: int
) -> tuple[float, float]:
    alpha_time = 0.0
    kmeans_time = 0.0
    if points.shape[1] == 2:
        ac = AlphaShape2D(points, alpha)
    else:
        ac = AlphaShape3D(points, alpha)
    for _ in range(runs):
        start = time.perf_counter()
        ac.fit()
        clustering = Cluster(
            shape = ac.get_shape
        )
        clustering.fit()
        clusters = list(clustering.predict_iter(4))
        alpha_time += time.perf_counter() - start

        start = time.perf_counter() 
        kmeans = KMeans(k_means)
        kmeans.fit_predict(points)
        kmeans_time += time.perf_counter() - start

    return alpha_time / runs, kmeans_time / runs

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
        alpha_time, kmeans_time = time_experiment(points, alpha, k_means, 10)
        df = pd.DataFrame(
            {
                "alpha clustering time": alpha_time,
                "kmeans clustering time": kmeans_time
            },
            index = [dataset_name]
        ) 
        dfs.append(df)

    df = pd.concat(dfs)

    io_handler.write_results(
        dataset = "all",
        results = dfs,
        save_name = "performance_results.tex",
        caption = "Performance results for all datasets.",
        join_axis = 0
    )

    t_stat, p_value = get_significance(
        df["alpha clustering time"],
        df["kmeans clustering time"]
    )
    print(f"t-stat: {t_stat}, p-value: {p_value}")

    return 0 

if __name__ == "__main__":
    import sys
    sys.exit(main())