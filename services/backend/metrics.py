import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np
import random

from neighbors import Neighbors, ComputedNeighbors, CachedNeighbors
from lmds import Lmds

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

        self.ld_neighbors = ComputedNeighbors(distance_metric=self.distance_metric, k=k, dimensions=Neighbors.DIMENSIONS_2D, dataset=data)
        trustworthiness, continuity = self.get_trustworthiness_and_continuity(k)
        metric =  {
            "trustworthiness": trustworthiness,
            "continuity": continuity,
            "neighborhood_hit": self.neighborhood_hit(k),
            "shepard_goodness": self.shepard_goodness(k),
            "average_local_error": self.average_local_error(k),
            "normalized_stress": self.normalized_stress(k),
        }
        self.metrics[k] = metric
        return metric


    def get_trustworthiness_and_continuity(self, k: int) -> Tuple[float, float]:
        # Get the rank matrix in the high and low dimensional space
        ld_rank = self.ld_neighbors.get_ranks()
        hd_rank = self.hd_neighbors.get_ranks()

        t_outer_sum = 0
        c_outer_sum = 0
        # In this formula the paper and code differ. The paper has a small n at (2*n-3*k-1). The code version was choosen.
        factor = 2/(self.N * k * (2*self.N - 3*k - 1))
        for i in range(self.N):
            ld_knn = [neighbor[0] for neighbor in self.ld_neighbors.get_k_neighbors(i)]
            hd_knn = [neighbor[0] for neighbor in self.hd_neighbors.get_k_neighbors(i)[:k]]
            hd_nn = next(hd_rank)
            ld_nn = next(ld_rank)

            # Again paper and code differ. The paper defines r as the rank the point j has in regards to i in the low dimensional space, while the code version uses the rank in the high dimensional space. The code version was choosen.
            U = set(ld_knn) - set(hd_knn)
            t_outer_sum += sum(hd_nn[j] - k for j in U)

            U_hat = set(hd_knn) - set(ld_knn)
            c_outer_sum += sum(ld_nn[j] - k for j in U_hat)

        return (1 - factor * t_outer_sum, 1 - factor * c_outer_sum)

    def neighborhood_hit(self, k: int) -> float:
        # Pseudocode: mean(mean(1 if label(j) == label(i) else 0 for j in neighbors(i)) for i in range(N)
        return np.mean(np.mean(1 if self.data.iloc[j]['label'] == self.data.iloc[i]['label'] else 0 for j in [neighbor[0] for neighbor in self.ld_neighbors.get_k_neighbors(i)]) for i in range(self.N))

    def shepard_diagram(self, k: int) -> List[Tuple[float]]:
        # TODO: The Shepard diagram is a scatterplot of the pairwise (euclidean) distances between all points in P(D) versus the corresponding distances in D. We would need all point pair distances in the high and low dimensional space
        return 0.5

    def shepard_diagram_plot(self, k: int) -> str:
        # scatterplot with matplotlib
        # TODO: See shepard_diagram
        return 0.5 #filehandle

    def shepard_goodness(self, k: int) -> float:
        # TODO: Does not make sense without shepard_diagram
        return 0.4

    def average_local_error(self, k: int) -> List[float]:
        # TODO: We would need all point pair distances in the high and low dimensional space
        return [random.random() for i in range(self.N)]

    def normalized_stress(self, k: int) -> float:
        # TODO: Problem we would need all point pair distances in the high and low dimensional space
        # Formula: np.sum((high_dist - low_dist)**2) / np.sum(high_dist**2)
        return 0.7

def wait_for_debugger(port: int = 56789):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using "
        f"docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()

if __name__ == '__main__':
    CachedNeighbors.ALL_NEIGHBORS_768D_FILENAME = './volumes/data/imdb_{distance_metric}_neighbors.bin'
    #wait_for_debugger()
    import pickle

    with open("./volumes/data/imdb_embeddings.pkl", "rb") as file:
        dataset = pickle.load(file)

    dataset_length = len(dataset)
    print(f"Dataset Length: {dataset_length}")
    print()

    lmds = Lmds(
        heuristic="random",
        distance_metric="euclidean",
        num_landmarks=10,
        dataset=dataset,
        do_pca=False,
    )

    lmds.select_landmarks()
    lmds.reduce_landmarks()
    print("Landmarks:")
    print(lmds.landmarks)
    print()

    lmds.calculate()
    dataset = lmds.all_points.sort_index()
    metrics = Metrics(dataset, "euclidean", 2)
    print(metrics.get_trustworthiness_and_continuity())