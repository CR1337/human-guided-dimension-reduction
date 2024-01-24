import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.stats import spearmanr

from neighbors import Neighbors, ComputedNeighbors, CachedNeighbors

# The metrics are based on: "Toward a Quantitative Survey of Dimension Reduction Techniques" (DOI: 10.1109/TVCG.2019.2944182)
# and their implementation in: https://github.com/mespadoto/proj-quant-eval/blob/master/code/01_data_collection/metrics.py

class Metrics:
    def __init__(self, distance_metric: str, use_small: bool) -> None:
        self.hd_neighbors = CachedNeighbors.all_neighbors_768d(distance_metric=distance_metric, use_small=use_small)
        self.distance_metric = distance_metric
        self.ld_neighbors = None
        self.data = None
        self.N = None
        self.metrics = {}


    def calculate_all_metrics(self, data: pd.DataFrame, k: int = 7) -> Dict[str, Any]:
        if k in self.metrics:
            return self.metrics[k]
        self.data = data
        self.N = len(data)

        self.ld_neighbors = ComputedNeighbors(distance_metric=self.distance_metric, k=self.N -1, dimensions=Neighbors.DIMENSIONS_2D, dataset=data)
        ld_dist, hd_dist = self.get_distance_matrices()
        ld_knn, hd_knn = self.get_k_nearest_neighbors(k)
        trustworthiness, continuity = self.get_trustworthiness_and_continuity(k, ld_knn, hd_knn)
        metric =  {
            "trustworthiness": trustworthiness,
            "continuity": continuity,
            "normalized_stress": self.normalized_stress(ld_dist, hd_dist),
            "neighborhood_hit": self.neighborhood_hit(ld_knn),
            # "shepard_goodness": self.shepard_goodness(ld_dist, hd_dist),
            "average_local_error": self.average_local_error(ld_dist, hd_dist),
        }
        self.metrics[k] = metric
        return metric


    def get_distance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function returns the distance matrices for the low and high dimensional space. The matrices are of the form: self.N * (self.N - 1)
        """
        ld_dist = np.asarray([[neighbor[1] for neighbor in self.ld_neighbors.get_k_neighbors(i)] for i in range(self.N)])
        hd_dist = np.asarray([[neighbor[1] for neighbor in self.hd_neighbors.get_k_neighbors(i)] for i in range(self.N)])
        return (ld_dist, hd_dist)
    
    def get_k_nearest_neighbors(self, k: int) -> (List[List[int]], List[List[int]]):
        """
        This function returns the k nearest neighbors for each point in the low dimensional space. The list is of the form: self.N * k
        """
        ld_knn = []
        hd_knn = []
        for i in range(self.N):
            ld_point_knn = []
            hd_point_knn = []
            for j, neighbors in enumerate(zip(self.ld_neighbors.get_k_neighbors(i), self.hd_neighbors.get_k_neighbors(i))):
                    if j >= k:
                        break
                    ld_point_knn.append(neighbors[0][0])
                    hd_point_knn.append(neighbors[1][0])
            ld_knn.append(ld_point_knn)
            hd_knn.append(hd_point_knn)
        return (ld_knn, hd_knn)


    def get_trustworthiness_and_continuity(self, k: int, ld_knn: List[List[int]], hd_knn: List[List[int]]) -> Tuple[float, float]:
        # Get the rank matrix in the high and low dimensional space
        ld_rank = self.ld_neighbors.get_ranks()
        hd_rank = self.hd_neighbors.get_ranks()

        t_outer_sum = 0
        c_outer_sum = 0
        # In this formula the paper and code differ. The paper has a small n at (2*n-3*k-1). The code version was choosen.
        factor = 2/(self.N * k * (2*self.N - 3*k - 1))
        for i in range(self.N):
            ld_point_knn = ld_knn[i]
            hd_point_knn = hd_knn[i]
            hd_nn = next(hd_rank)
            ld_nn = next(ld_rank)

            # Again paper and code differ. The paper defines r as the rank the point j has in regards to i in the low dimensional space, while the code version uses the rank in the high dimensional space. The code version was choosen.
            U = set(ld_point_knn) - set(hd_point_knn)
            t_outer_sum += sum(hd_nn[j] - k for j in U)

            U_hat = set(hd_point_knn) - set(ld_point_knn)
            c_outer_sum += sum(ld_nn[j] - k for j in U_hat)

        return (1 - factor * t_outer_sum, 1 - factor * c_outer_sum)


    def normalized_stress(self, ld_dist: np.ndarray, hd_dist: np.ndarray) -> float:
        return np.sum((hd_dist - ld_dist)**2) / np.sum(hd_dist**2)

    def neighborhood_hit(self, ld_knn: List[List[int]]) -> float:
        labels = list(self.data['label'])
        # Pseudocode: mean(mean(1 if label(j) == label(i) else 0 for j in neighbors(i)) for i in range(N)
        return np.mean([np.mean([1 if labels[j] == labels[i] else 0 for j in ld_knn[i]]) for i in range(self.N)])

    def shepard_diagram(self, ld_dist: np.ndarray, hd_dist: np.ndarray) -> List[Tuple[float]]:
        return 0.5

    def shepard_diagram_plot(self, k: int) -> str:
        # scatterplot with matplotlib
        # TODO: See shepard_diagram
        return 0.5 #filehandle

    def shepard_goodness(self, ld_dist: np.ndarray, hd_dist: np.ndarray) -> float:
        return spearmanr(ld_dist, hd_dist)[0]

    def average_local_error(self, ld_dist: np.ndarray, hd_dist: np.ndarray) -> List[float]:
        # Averaged sum of difference normalized distances between the low and high dimensional space
        return [np.mean(np.abs(ld_dist[i] / max(ld_dist[i]) - hd_dist[i] / max(hd_dist[i]))) for i in range(self.N)]
