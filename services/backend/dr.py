import pandas as pd
import numpy as np
import itertools
from random import Random
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from typing import Any, Callable, Dict, List, Tuple

from metrics import Metrics
from dataset import Dataset
from idr import InverseDimensionaltyReduction


def balanced_heuristic(
    dataset: Dataset, num_landmarks: int, seed: int
) -> pd.DataFrame:
    df = dataset.dataframe.sample(frac=1, random_state=seed)
    label_dfs = [df[df["label"] == label] for label in df["label"].unique()]
    Random(seed).shuffle(label_dfs)

    return pd.concat(
        [
            label_df.iloc[i % num_landmarks].to_frame().T
            for i, label_df in zip(
                range(num_landmarks), itertools.cycle(label_dfs)
            )
        ]
    )


class DimensionalityReduction:
    METHODS: List[str] = ["MDS", "t-SNE"]
    HEURISTICS: List[str] = ["balanced", "random", "first"]
    DISTANCE_METRICS: List[str] = ["euclidean", "cosine"]
    LANDMARK_AMOUNT_RANGE: Tuple[int, int] = (10, 30)

    _heuristic: str
    _distance_metric: str
    _num_landmarks: int
    _dataset: pd.DataFrame
    _dimension: int
    _method: str

    _heuristic_func: Callable
    _distance_metric_func: Callable

    _landmarks: pd.DataFrame | None
    _no_landmark_points: pd.DataFrame | None

    _landmark_embeddings: np.ndarray | None
    _delta_n: np.ndarray | None
    _eigenvalues: np.ndarray | None
    _eigenvectors: np.ndarray | None

    _landmarks_reduced: bool
    _points_calculated: bool

    _metrics: Metrics

    _last_idr_algorithm: str | None

    def __init__(
        self,
        heuristic: str,
        distance_metric: str,
        num_landmarks: int,
        dataset: Dataset,
        dimension: int = 2,
        create_dataset: bool = False,
        method: str = "MDS",
    ):
        self._heuristic = heuristic
        if heuristic == "random":
            self._heuristic_func = (
                lambda dataset, num_landmarks, seed: dataset.sample(
                    n=num_landmarks, random_state=seed
                )
            )
        elif heuristic == "first":
            self._heuristic_func = (
                lambda dataset, num_landmarks, _: dataset.dataframe.head(
                    num_landmarks
                )
            )
        elif heuristic == "balanced":
            self._heuristic_func = balanced_heuristic
        else:
            raise NotImplementedError(f"Unknown heuristic: {heuristic}")

        self._distance_metric = distance_metric
        if distance_metric == "euclidean":
            self._distance_metric_func = euclidean_distances
        elif distance_metric == "cosine":
            self._distance_metric_func = cosine_distances
        else:
            raise NotImplementedError(
                f"Unknown distance metric: {distance_metric}"
            )

        self._num_landmarks = num_landmarks
        self._dataset = dataset
        self._dimension = dimension
        self._method = method

        self._landmarks = None
        self._no_landmark_points = None

        self._landmark_embeddings = None
        self._delta_n = None
        self._eigenvalues = None
        self._eigenvectors = None

        self._landmarks_reduces = False
        self._points_calculated = False

        if not create_dataset:
            self._metrics = Metrics(
                distance_metric, dataset.neighbors(distance_metric)
            )

        self._last_idr_algorithm = None

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def landmarks(self) -> pd.DataFrame:
        return self._landmarks

    @landmarks.setter
    def landmarks(self, landmarks: pd.DataFrame):
        self._landmarks = landmarks.copy()

    @property
    def no_landmark_points(self) -> pd.DataFrame:
        return self._no_landmark_points

    @property
    def all_points(self):
        if self._landmarks is None:
            raise RuntimeError("Landmarks not selected!")
        if self._no_landmark_points is None:
            raise RuntimeError("Points not computed!")
        return pd.concat([self._landmarks, self._no_landmark_points])

    @property
    def landmarks_selected(self) -> bool:
        return self._landmarks is not None

    @property
    def landmarks_reduced(self) -> bool:
        return self._landmarks_reduced

    @property
    def points_calculated(self) -> bool:
        return self._points_calculated

    @property
    def high_landmark_embeddings(self) -> np.array:
        """
        Returns the high dimensional embeddings of the landmarks.
        """
        return np.vstack(self._landmarks["embeddings"].apply(np.array))

    @property
    def low_landmark_embeddings(self) -> np.array:
        """
        Returns the 2D embeddings of the landmarks.
        """
        return np.vstack(self._landmarks["position"].apply(np.array))

    def distances(self, vector1: np.array, vector2: np.array) -> np.ndarray:
        return self._distance_metric_func(vector1, vector2)

    def compute_metrics(self, k: int) -> Dict[str, Any]:
        if not self.points_calculated:
            raise RuntimeError("Points not calculated!")
        return self._metrics.calculate_all_metrics(
            self.all_points,
            self._last_idr_algorithm,
            hash(str(self._landmarks['position'])),
            k
        )

    def select_landmarks(self, seed: int = 42):
        self._landmarks = self._heuristic_func(
            self._dataset, self._num_landmarks, seed
        )

    def reduce_landmarks(self):
        if not self.landmarks_selected:
            raise RuntimeError("Landmarks not selected!")
        # Deltan is the squared distance matrix between the landmarks
        self._delta_n = (
            self._distance_metric_func(
                self.high_landmark_embeddings, self.high_landmark_embeddings
            )
            ** 2
        )
        # To cache the distance matrix
        self._delta_n_old = self._delta_n

        # compute eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors = self._compute_eigenstuff()
        if self._method == "MDS":
            # We compute the matrix L which is given
            # by self._eigenvectors * sqrt(self._eigenvalues)
            pos_eigenvalues = self._eigenvalues[self._eigenvalues > 0]
            if len(pos_eigenvalues) < self._dimension:
                print(
                    "Error: Not enough positive eigenvalues "
                    f"for the selected dimension {self._dimension}."
                )
                return []
            self._L = np.zeros((len(self._landmarks), self._dimension))
            for i in range(self._dimension):
                self._L[:, i] = self._eigenvectors[:, i] * np.sqrt(
                    self._eigenvalues[i]
                )
        elif self._method == "t-SNE":
            from sklearn.manifold import TSNE
            tsne = TSNE(
                n_components=self._dimension,
                metric="precomputed",
                perplexity=7,
                init="random",
                random_state=42
            )
            self._L = tsne.fit_transform(self._delta_n)
        else:
            raise NotImplementedError(f"Unknown method: {self._method}")

        # Append the position of the landmarks to the dataset
        self._landmarks = self._landmarks.assign(
            position=self._L.tolist(), landmark=True
        )

        self._landmarks_reduced = True

    def calculate(self, idr_algorithm: str):
        if not self.landmarks_reduced:
            raise RuntimeError("Landmarks not reduced!")

        # Compute new delta_n using one of the inverse dr algorithms
        low_dimensional_distances = self._distance_metric_func(
            self.low_landmark_embeddings, self.low_landmark_embeddings
        )
        self._delta_n = InverseDimensionaltyReduction(
            idr_algorithm, self._distance_metric, self._method
        ).inference(low_dimensional_distances, self._delta_n_old)

        # recompute eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors = self._compute_eigenstuff()

        # The mean distance between the landmarks
        # is the mean of the columns of Deltan
        mean_distance = self._delta_n.mean(axis=0)

        # L_sharp is the pseudo-inverse of L
        # given by eigenvectors * 1/sqrt(eigenvalues)
        L_sharp = np.zeros((self._dimension, len(self._landmarks)))
        for i in range(self._dimension):
            L_sharp[i, :] = (
                self._eigenvectors[:, i].transpose() * 1 / np.sqrt(
                    self._eigenvalues[i]
                )
            )

        # We first need to get the embeddings of the other points
        other_points = self._dataset.dataframe[
            ~self._dataset.dataframe.index.isin(self._landmarks.index)
        ]
        other_embeddings = np.vstack(
            other_points["embeddings"].apply(np.array)
        )

        # We compute for each point the distance to the landmarks
        distance_to_landmarks = (
            self._distance_metric_func(
                other_embeddings, self.high_landmark_embeddings
            ) ** 2
        )

        # Going through each point, we compute its position
        # by -1/2 * L_sharp * (distance_to_landmarks - mean_distance)
        positions = np.zeros((len(other_points), self._dimension))
        for i in range(len(other_points)):
            position = -1 / 2 * (
                L_sharp.dot(distance_to_landmarks[i] - mean_distance)
            )
            positions[i, :] = position

        # Append the position of the other points to the dataset
        self._no_landmark_points = other_points.assign(
            position=positions.tolist(), landmark=False
        )

        self._points_calculated = True
        self._last_idr_algorithm = idr_algorithm

    def _compute_eigenstuff(self) -> Tuple[np.ndarray, np.ndarray]:
        # H is the mean centering matrix
        H = -np.ones(
            (self._num_landmarks, self._num_landmarks)
        ) / self._num_landmarks
        np.fill_diagonal(H, 1 - 1 / self._num_landmarks)

        # B is the mean centered "inner-product" matrix
        B = -1 / 2 * (H.dot(self._delta_n).dot(H))

        # We compute the eigenvalues and eigenvectors of B
        eigenvalues, eigenvectors = np.linalg.eigh(B)

        # We sort the eigenvalues and eigenvectors by decreasing eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def to_json(self) -> Dict[str, Any]:
        return {
            "heuristic": self._heuristic,
            "distance_metric": self._distance_metric,
            "num_landmarks": self._num_landmarks,
            "landmarks_selected": self.landmarks_selected,
            "landmarks_reduced": self.landmarks_reduced,
            "points_calculated": self.points_calculated,
            "labels": self._dataset.labels
        }
