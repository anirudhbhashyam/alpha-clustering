import numpy as np

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