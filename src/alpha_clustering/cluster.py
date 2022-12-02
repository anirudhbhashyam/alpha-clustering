import collections
from dataclasses import dataclass, field

from typing import Iterator

# import networkx as nx

import numpy as np
import pandas as pd

from sklearn import metrics

from .logger import Logger
from .exceptions import NotFittedError

LOGGER = Logger(__name__)

@dataclass
class Cluster:
    """
    Custom interface to cluster data using an Alpha complex.
    """
    shape: list[np.ndarray]

    _adjacency_list: dict[int, set[int]] = field(
        default_factory = lambda: collections.defaultdict(set)
    )

    clusters: list[set[int]] = field(
        default_factory = list
    )
    
    def fit(self) -> None:
        """
        Constructs an intermediary data structure from the Alpha complex.
        """
        LOGGER.info("Fitting the cluster model...")
        edges = self.shape[-1]

        for v1, v2 in edges:
            self._adjacency_list[v1].add(v2)
            self._adjacency_list[v2].add(v1)
                        
    def predict_iter(self, cluster_threshold: int) -> Iterator[set[int]]:
        """
        Predicts clusters using a connected component search.

        Parameters
        ----------
        cluster_threshold: 
            The minimum number of points that should be present for a set to be considered a cluster.

        Returns
        -------
        ``Iterator[set[int]]``:
            Iterable that yields clusters.
        """
        LOGGER.info("Predicting the clusters using the \u03B1-complex...(iterating)")
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
        """
        Predicts clusters using a connected component search.

        Parameters
        ----------
        cluster_threshold: 
            The minimum number of points that should be present for a set to be considered a cluster.

        Returns
        -------
        ``list[list[int]]``:
            A list of clusters. Each sublist is a cluster containing the indices of the vertices.
        """
        LOGGER.info("Predicting the clusters using the \u03B1-complex...")
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

    # def _find_communities(self) -> list[set[int]]:
    #     LOGGER.info("Finding the communities...")
    #     graph = nx.Graph(self._adjacency_list)
    #     communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    #     m = nx.algorithms.community.modularity(graph, communities)
    #     return m, communities
        
    def _bfs(self, source: float) -> set:
        """
        A private function to perform a breadth first search used by `self.predict`.

        Parameters
        ----------
        source: 
            A node in the graph from where a bfs should be done.

        Returns
        -------
        set:
            The visited nodes from the source after a bfs is conducted.
        """
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
        """
        A convenience method that is used to summarise a clustering task.
        
        Parameters
        ----------
        dataset: 
            The name of the dataset that has been clustered.
        other_cluster_data: 
            Key value pairs of metadata of any other clustering method that has been used.
        threshold:
            The clustering threshold used for alpha clustering.
        """
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
            columns = [f"Dataset: {dataset}"]
        )

@dataclass
class ClusterEvaluate:
    """
    A simple class that evaluates cluster labels using different metrics. Should be used to compare the performance a clustering method. 
    """
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
        """
        Performs an external evaluation of cluster labels. The external evaluation will utilise the ground truth clusters.

        Parameters
        ----------
        method:
            The clustering method used.
        """
        LOGGER.info(f"Evaluating the clusters predicted by method {method} using external data...")
        self.rand_score = metrics.rand_score(self.true_clusters, self.predicted_clusters)
        self.adjusted_rand_score = metrics.adjusted_rand_score(self.true_clusters, self.predicted_clusters)
        self.mutual_info_score = metrics.mutual_info_score(self.true_clusters, self.predicted_clusters)
        self.homogeneity_score = metrics.homogeneity_score(self.true_clusters, self.predicted_clusters)

    def _internal_evaluation(self, method: str) -> None:
        """
        Performs an internal evaluation of cluster labels.

        Parameters
        ----------
        method:
            The clustering method used.
        """
        LOGGER.info(f"Evaluating the clusters predicted by method {method} using internal metrics...")
        self.silhouette_score = metrics.silhouette_score(self.data, self.predicted_clusters)
        self.davies_bouldin_score = metrics.davies_bouldin_score(self.data, self.predicted_clusters)

    def get_results(self, method: str) -> pd.DataFrame:
        """
        Summarises the results from both types of evaluations.

        Parameters
        ----------
        method:
            The clustering method to which alpha clustering is compared to.

        Returns
        -------
        ``pandas DataFrame``:
            A dataframe containing all evaluation results.
        """
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

