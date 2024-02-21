import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from typing import Any, Callable, Dict, List, Tuple
from functools import cached_property

from metrics import Metrics
from neighbors import CachedNeighbors


def random_landmarks_heuristic(
    dataset: pd.DataFrame, num_landmarks: int, seed: int
) -> pd.DataFrame:
    return dataset.sample(n=num_landmarks, random_state=seed)


class Lmds:

    HEURISTICS: List[str] = ["random", "first"]
    DISTANCE_METRICS: List[str] = ["euclidean", "cosine"]
    LANDMARK_AMOUNT_RANGE: Tuple[int, int] = (10, 30)
    IMDS_ALGORITHMS: List[str] = ["trivial"]

    _heuristic: str
    _distance_metric: str
    _num_landmarks: int
    _dataset: pd.DataFrame
    _dimension: int

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

    def __init__(
        self,
        heuristic: str,
        distance_metric: str,
        num_landmarks: int,
        dataset: pd.DataFrame,
        dimension: int = 2,
        use_small: bool = True,
        debug: bool = False,
    ):
        if debug:
            wait_for_debugger()
            CachedNeighbors.ALL_NEIGHBORS_768D_FILENAME = (
                './volumes/data/imdb_{distance_metric}_neighbors_small.bin'
            )

        self._heuristic = heuristic
        if heuristic == "random":
            self._heuristic_func = random_landmarks_heuristic
        elif heuristic == "first":
            self._heuristic_func = lambda dataset, num_landmarks: dataset.head(num_landmarks)
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

        self._landmarks = None
        self._no_landmark_points = None

        self._landmark_embeddings = None
        self._delta_n = None
        self._eigenvalues = None
        self._eigenvectors = None

        self._landmarks_reduces = False
        self._points_calculated = False

        self._metrics = Metrics(distance_metric, use_small)

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def landmarks(self) -> pd.DataFrame:
        return self._landmarks

    @landmarks.setter
    def landmarks(self, landmarks: pd.DataFrame):
        self._landmarks = landmarks

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

    @cached_property
    def high_landmark_embeddings(self) -> np.array:
        """
        Returns the high dimensional embeddings of the landmarks.
        """
        return np.vstack(self._landmarks['embeddings'].apply(np.array))

    @cached_property
    def low_landmark_embeddings(self) -> np.array:
        """
        Returns the 2D embeddings of the landmarks.
        """
        return np.vstack(self._landmarks['position'].apply(np.array))

    def compute_metrics(self, k: int) -> Dict[str, Any]:
        if not self.points_calculated:
            raise RuntimeError("Points not calculated!")
        return self._metrics.calculate_all_metrics(self.all_points, k)


    def select_landmarks(self, seed: int = 42):
        self._landmarks = self._heuristic_func(self._dataset, self._num_landmarks, seed)

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

        # compute eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors = self._compute_eigenstuff()

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

        # Append the position of the landmarks to the dataset
        self._landmarks = self._landmarks.assign(
            position=self._L.tolist(), landmark=True
        )

        self._landmarks_reduced = True

    def calculate(self, imds_algorithm: str, do_pca: bool):
        if not self.landmarks_reduced:
            raise RuntimeError("Landmarks not reduced!")

        # Compute new delta_n using one of the imds algorithms
        if imds_algorithm == 'trivial':
            # Just use the low dimensional distances
            # as new high dimensional delta_n
            self._delta_n = (
                self._distance_metric_func(
                    self.low_landmark_embeddings,
                    self.low_landmark_embeddings
                )
                ** 2
            )
        else:
            raise RuntimeError(f"Unknown imds algorithm: {imds_algorithm}")

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
        other_points = self._dataset[
            ~self._dataset.index.isin(self._landmarks.index)
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

        if do_pca:
            positions = self._pca(positions)

        # Append the position of the other points to the dataset
        self._no_landmark_points = other_points.assign(
            position=positions.tolist(), landmark=False
        )

        self._points_calculated = True

    def _pca(self, positions: np.ndarray) -> np.ndarray:
        # We merge the positions of the landmarks and the other points
        X = np.vstack([self._L, positions]).T
        # We compute the mean of each dimension
        X_mean = np.mean(X, axis=1)

        # We center the data
        # X_hat = X - X_mean
        X_hat = np.zeros((self._dimension, X.shape[1]))
        for i in range(X.shape[1]):
            X_hat[:, i] = X[:, i] - X_mean[:]

        # We compute the eigenvalues and eigenvectors of X_hat * X_hat.T
        __, eigenvectors = np.linalg.eigh(X_hat.dot(X_hat.T))

        # With the eigenvectors we can compute the new positions
        X_new = eigenvectors.T.dot(X_hat).T

        self._landmarks = self._landmarks.assign(
            position=X_new[: len(self._landmarks), :].tolist(), landmark=True
        )
        return X_new[len(self._landmarks):, :]

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
        }


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


if __name__ == "__main__":
    import pickle

    with open("./volumes/data/imdb_embeddings_small.pkl", "rb") as file:
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
        debug=True,
    )

    lmds.select_landmarks()
    lmds.reduce_landmarks()
    print("Landmarks:")
    print(lmds.landmarks)
    print()

    lmds.calculate()
    print("All points:")
    print(lmds.all_points)
    metrics = lmds.compute_metrics(7)
    print(metrics)
