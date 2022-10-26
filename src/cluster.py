import collections
from dataclasses import dataclass, field

import trimesh

import numpy as np
import pandas as pd

from sklearn import metrics

from exceptions import NotFittedError
from logger import Logger

LOGGER = Logger(__name__)

@dataclass
class Cluster:
    shape: tuple[np.ndarray]

    _adjacency_list: dict[int, set[int]] = field(
        default_factory = lambda: collections.defaultdict(set)
    )

    clusters: list[set[int]] = field(
        default_factory = list
    )
    
    def fit(self) -> None:
        LOGGER.info("Fitting the cluster model...")
        if isinstance(self.shape, trimesh.Trimesh):
            edges = trimesh.geometry.faces_to_edges(self.shape.faces)
        else:
            edges = self.shape[-1]

        for v1, v2 in edges:
            self._adjacency_list[v1].add(v2)
            self._adjacency_list[v2].add(v1)
                        
    def predict_iter(self, cluster_threshold: int) -> set:
        LOGGER.info("Predicting the clusters using the \u03B1-shape...(iterating)")
        if len(self._adjacency_list) == 0:
            raise NotFittedError(
                self.__class__.__name__,
                LOGGER
            )
        visited = set()
        for node in self._adjacency_list:
            if node not in visited:
                component = self._bfs(node)
                visited.update(component)
                if len(component) > cluster_threshold:
                    yield component

    def predict(self, cluster_threshold: int = 2) -> list[list[int]]:
        LOGGER.info("Predicting the clusters using the \u03B1-shape...")
        # if len(self._adjacency_list) == 0:
        #     raise NotFittedError(
        #         self.__class__.__name__,
        #         LOGGER
        #     )
        visited = set()
        for node in self._adjacency_list:
            if node not in visited:
                component = self._bfs(node)
                visited.update(component)
                if len(component) > cluster_threshold:
                    self.clusters.append(component)
        return self.clusters
        
    def _bfs(self, source: float) -> set:
        visited = set()
        queue = {source}
        while queue:
            current_graph_level = queue
            queue = set()
            for node in current_graph_level:
                if node in visited:
                    continue
                visited.add(node)
                queue.update(self._adjacency_list[node])
        return visited

    def get_summary(
        self, 
        dataset: str,
        other_cluster_data: list[tuple[str, int]],
        threshold: int
    ) -> pd.DataFrame:
        other_data_dict = {
            s: n for s, n in other_cluster_data
        }
        summary = {
            "Set cluster threshold": int(threshold),
            "Number of alpha clusters": len(self.clusters)
        }
        summary.update(other_data_dict) 
        return pd.DataFrame.from_dict(
            summary, 
            orient = "index",
            columns = [dataset]
        )

@dataclass
class ClusterEvaluate:
    data: np.ndarray
    true_clusters: np.array
    predicted_clusters: np.array
    rand_score: float = -100
    adjusted_rand_score: float = -100
    mutual_info_score: float = -100
    homogeneity_score: float = -100
    silhouette_score: float = -100
    davies_bouldin_score: float = -100

    def _external_evaluation(self, method: str) -> None:
        LOGGER.info(f"Evaluating the clusters predicted by method {method} using external data...")
        self.rand_score = metrics.rand_score(self.true_clusters, self.predicted_clusters)
        self.adjusted_rand_score = metrics.adjusted_rand_score(self.true_clusters, self.predicted_clusters)
        self.mutual_info_score = metrics.mutual_info_score(self.true_clusters, self.predicted_clusters)
        self.homogeneity_score = metrics.homogeneity_score(self.true_clusters, self.predicted_clusters)

    def _internal_evaluation(self, method: str) -> None:
        LOGGER.info(f"Evaluating the clusters predicted by method {method} using internal metrics...")
        self.silhouette_score = metrics.silhouette_score(self.data, self.predicted_clusters)
        self.davies_bouldin_score = metrics.davies_bouldin_score(self.data, self.predicted_clusters)

    def get_results(self, method: str) -> pd.DataFrame:
        LOGGER.info("Extracting the cluster evaluation results...")
        self._external_evaluation(method)
        self._internal_evaluation(method)

        results_dct = {
            "Rand Score": self.rand_score,
            "Adjusted Rand Score": self.adjusted_rand_score,
            "Mutual Information Score": self.mutual_info_score,
            "Homogeneity Score": self.homogeneity_score,
            "Silhouette Score": self.silhouette_score,
            "Davies Bouldin Score": self.davies_bouldin_score
        }
            
        return pd.DataFrame.from_dict(
            results_dct,
            orient = "index", 
            columns = [method]
        )

    def _save(self) -> None:
        raise NotImplementedError(f"Save functionality for {self.__class__.__name__} not implemented.")

