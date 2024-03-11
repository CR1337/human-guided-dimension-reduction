from __future__ import annotations
import os
import json
import pickle
import pandas as pd
from typing import Any, List, Dict

from neighbors import CachedNeighbors


class Dataset:

    VALID_NAMES: List[str] = ["imdb_small", "imdb", "emotion"]

    DOCKER_PATH: str = "/server/data"
    LOCAL_PATH: str = "./volumes/data"

    _name: str
    _no_neighbors: bool

    _dataset_path: str
    _cosine_neighbors_path: str
    _euclidean_neighbors_path: str
    _metadata_path: str

    _dataframe: pd.DataFrame
    _cosine_neighbors: CachedNeighbors
    _euclidean_neighbors: CachedNeighbors
    _metadata: Dict[str, Any]

    @classmethod
    def all(cls, no_neighbors: bool = True) -> List[Dataset]:
        return [cls(name, no_neighbors) for name in cls.VALID_NAMES]

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def cosine_neighbors_path(self) -> str:
        return self._cosine_neighbors_path

    @property
    def euclidean_neighbors_path(self) -> str:
        return self._euclidean_neighbors_path

    @property
    def metadata_path(self) -> str:
        return self._metadata_path

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def cosine_neighbors(self) -> CachedNeighbors | None:
        if self._no_neighbors:
            return None
        return self._cosine_neighbors

    @property
    def euclidean_neighbors(self) -> CachedNeighbors | None:
        if self._no_neighbors:
            return None
        return self._euclidean_neighbors

    @property
    def labels(self) -> List[str]:
        return self._metadata['labels']

    def neighbors(self, distance_metric: str) -> CachedNeighbors:
        if distance_metric == 'cosine':
            return self.cosine_neighbors
        elif distance_metric == 'euclidean':
            return self.euclidean_neighbors
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

    def __init__(self, name: str, no_neighbors: bool = False):
        if name not in self.VALID_NAMES:
            raise ValueError(f"Invalid dataset name: {name}")
        self._name = name
        self._no_neighbors = no_neighbors

        inside_docker = bool(os.environ.get('INSIDE_DOCKER', False))

        self._dataset_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_embeddings.pkl"
        )
        self._cosine_neighbors_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_cosine_neighbors.bin"
        )
        self._euclidean_neighbors_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_euclidean_neighbors.bin"
        )
        self._metadata_path = os.path.join(
            self.DOCKER_PATH if inside_docker else self.LOCAL_PATH,
            f"{self._name}_meta.json"
        )

        with open(self._dataset_path, 'rb') as file:
            self._dataframe = pickle.load(file)
        if not self._no_neighbors:
            self._cosine_neighbors = CachedNeighbors(
                self._cosine_neighbors_path
            )
            self._euclidean_neighbors = CachedNeighbors(
                self._euclidean_neighbors_path
            )
        with open(self._metadata_path, 'r') as file:
            self._metadata = json.load(file)
