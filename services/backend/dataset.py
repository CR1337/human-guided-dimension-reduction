import os
import pickle
import pandas as pd
from typing import List

from neighbors import CachedNeighbors


class Dataset:

    VALID_NAMES: List[str] = ["imdb_small", "imdb", "emotions"]

    DOCKER_PATH: str = "/server/data"
    LOCAL_PATH: str = "./volumes/data"

    _name: str
    _inside_docker: bool

    _dataframe: pd.DataFrame
    _cosine_neighbors: CachedNeighbors
    _euclidean_neighbors: CachedNeighbors

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def cosine_neighbors(self) -> CachedNeighbors:
        return self._cosine_neighbors

    @property
    def euclidean_neighbors(self) -> CachedNeighbors:
        return self._euclidean_neighbors

    def __init__(self, name: str):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Invalid dataset name: {name}")
        self._name = name

        inside_docker = bool(os.environ.get('INSIDE_DOCKER', False))

        dataset_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_embeddings.pkl"
        )
        cosine_neighbor_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_cosine_neighbors.bin"
        )
        euclidean_neighbor_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_euclidean_neighbors.bin"
        )

        self._dataframe = pickle.load(dataset_path)
        self._cosine_neighbors = CachedNeighbors(cosine_neighbor_path)
        self._euclidean_neighbors = CachedNeighbors(euclidean_neighbor_path)
