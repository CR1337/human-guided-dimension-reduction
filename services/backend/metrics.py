import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import heapq
import numpy as np

from neighbors import Neighbors
from lmds import Lmds

# The metrics are based on: "Toward a Quantitative Survey of Dimension Reduction Techniques" (DOI: 10.1109/TVCG.2019.2944182)
# and their implementation in: https://github.com/mespadoto/proj-quant-eval/blob/master/code/01_data_collection/metrics.py

def _get_high_dimensional_neighbors(point_index: int, distance_metric: str, k: int) -> List[Tuple[int, float]]:
    return Neighbors.get(point_index=point_index, k=k, distance_metric=distance_metric)

def _get_low_dimensional_neighbors(point_index:int, data: pd.DataFrame, distance_metric: str, k: int) -> List[Tuple[int, float]]:
    if distance_metric == "euclidean":
        distance_metric_func = euclidean_distances
    elif distance_metric == "cosine":
        distance_metric_func = cosine_distances

    # We need to convert the data to a numpy array, because the distance_metric_func needs a 2D array
    neighbors = heapq.nsmallest(k + 1, [(distance_metric_func(np.array(data.iloc[point_index]).reshape(1,-1), np.array(data.iloc[i]).reshape(1,-1)), i) for i in range(len(data))]) # This function requires ca. 5 seconds per example
    return [i for __, i in neighbors if i != point_index]


def trustworthiness(data: pd.DataFrame, distance_metric: str, k: int = 7) -> float:
    N = len(data)
    
    outer_sum = 0
    for i in range(N):
        hd_neighbors = _get_high_dimensional_neighbors(i, distance_metric, k)
        hd_neighbors = [n[0] for n in hd_neighbors]
        ld_neighbors = _get_low_dimensional_neighbors(i, data, distance_metric, k)
        inner_sum = 0
        for j, ld_n in enumerate(ld_neighbors):
            if ld_n not in hd_neighbors:
                inner_sum += j - k # TODO: After reviewing the code: It should be the index of the neighbor in the high dimensional space
        outer_sum += inner_sum
    # In this formula the paper and code differ. The paper has a small n at (2*n-3*k-1). The code version was choosen.
    return 1 - 2/(N * k * (2*N - 3*k - 1)) * outer_sum

def continuity(data: pd.DataFrame, distance_metric: str, k: int = 7) -> float:
    N = len(data)
    outer_sum = 0
    for i in range(N):
        hd_neighbors = _get_high_dimensional_neighbors(i, distance_metric, k)[:,0]
        ld_neighbors = _get_low_dimensional_neighbors(i, data, distance_metric, k)[:,0]
        inner_sum = 0
        for j, hd_n in enumerate(hd_neighbors):
            if hd_n not in ld_neighbors:
                inner_sum += j - k
        outer_sum += inner_sum
    # In this formula the paper and code differ. The paper has a small n at (2*n-3*k-1). The code version was choosen.
    return 1 - 2/(N * k * (2*N - 3*k - 1)) * outer_sum

def normalized_stress(data: pd.DataFrame, distance_metric: str) -> float:
    return 0.7

def neighborhood_hit(data: pd.DataFrame, distance_metric: str, k: int = 7) -> float:
    return 0.6

def shepard_diagram(data: pd.DataFrame, distance_metric: str) -> List[Tuple[float]]:
    # scatterplot with matplotlib
    return 0.5

def shepard_diagram_plot(data: List[Tuple[float]]) -> str:
    # scatterplot with matplotlib
    return 0.5 #filehandle

def shepard_goodness(data: List[Tuple[float]]) -> float:
    return 0.4

def average_local_error(data: pd.DataFrame, distance_metric: str) -> List[float]:
    return 0.3

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
    Neighbors.FILENAME = './volumes/data/imdb_{distance_metric}_neighbors.bin'
    # wait_for_debugger()
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
    dataset = lmds.all_points["position"].sort_index()
    # points = np.array([np.array(point) for point in dataset])
    # distance = euclidean_distances(points) # This function gets killed on my machine

    trustworthiness(dataset, "euclidean")