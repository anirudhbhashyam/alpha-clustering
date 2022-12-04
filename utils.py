import numpy as np
import pandas as pd

from typing import Callable, Any

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

def change_style_format_df(
    df: pd.DataFrame, 
    formatter: Callable[[Any], Any],
    row: str = None, 
    col: str = None 
) -> None:
    if row is not None:
        df.loc[row, :] = df.loc[row, :].apply(formatter)
    if col is not None:
        df.loc[:, col] = df.loc[:, col].apply(formatter)
