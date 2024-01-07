import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from typing import Any, Callable, Dict, List


def random_landmarks_heuristic(
    dataset: pd.DataFrame, num_landmarks: int
) -> pd.DataFrame:
    return dataset.sample(n=num_landmarks)


class Lmds:

    HEURISTICS: List[str] = ['random']
    DISTANCE_METRICS: List[str] = ['euclidean', 'cosine']

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

    def __init__(
        self,
        heuristic: str,
        distance_metric: str,
        num_landmarks: int,
        dataset: pd.DataFrame,
        dimension: int = 2
    ):
        self._heuristic = heuristic
        if heuristic == 'random':
            self._heuristic_func = random_landmarks_heuristic
        else:
            raise NotImplementedError(f"Unknown heuristic: {heuristic}")

        self._distance_metric = distance_metric
        if distance_metric == 'euclidean':
            self._distance_metric_func = euclidean_distances
        elif distance_metric == 'cosine':
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

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def landmarks(self) -> pd.DataFrame:
        return self._landmarks

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
        return self._landmark_embeddings is not None

    @property
    def points_calculated(self) -> bool:
        return self._no_landmark_points is not None

    def select_landmarks(self):
        self._landmarks = self._heuristic_func(
            self._dataset, self._num_landmarks
        )

    def reduce_landmarks(self):
        if not self.landmarks_selected:
            raise RuntimeError("Landmarks not selected!")

        # We need to transform landmarks["embeddings"] into a numpy array
        # with the shape (num_landmarks, embedding_dim)
        self._landmark_embeddings = np.vstack(
            self._landmarks['embeddings'].apply(np.array)
        )

        # Deltan is the distance matrix between the landmarks
        self._delta_n = self._distance_metric_func(
            self._landmark_embeddings, self._landmark_embeddings
        )
        # H is the mean centering matrix
        H = self._delta_n - 1 / self._num_landmarks
        # B is the mean centered "inner-product" matrix
        B = -1/2 * H @ self._delta_n @ H
        # We compute the eigenvalues and eigenvectors of B
        self._eigenvalues, self._eigenvectors = np.linalg.eig(B)
        # We sort the eigenvalues and eigenvectors by decreasing eigenvalues
        idx = self._eigenvalues.argsort()[::-1]
        self._eigenvalues = self._eigenvalues[idx]
        self._eigenvectors = self._eigenvectors[:, idx]
        # We compute the matrix L which is given
        # by self._eigenvectors * sqrt(self._eigenvalues)
        L = np.zeros((len(self._landmarks), self._dimension))
        for i in range(self._dimension):
            L[:, i] = self._eigenvectors[:, i] * np.sqrt(self._eigenvalues[i])

        # Append the position of the landmarks to the dataset
        self._landmarks = self._landmarks.assign(
            position=L.tolist(), landmark=True
        )

    def calculate(self):
        if not self.landmarks_reduced:
            raise RuntimeError("Landmarks not reduced!")

        # The mean distance between the landmarks
        # is the mean of the columns of Deltan
        mean_distance = self._delta_n.mean(axis=0)

        # L_sharp is the pseudo-inverse of L
        # given by eigenvectors * 1/sqrt(eigenvalues)
        L_sharp = np.zeros((self._dimension, len(self._landmarks)))
        for i in range(self._dimension):
            L_sharp[i, :] = (
                self._eigenvectors[:, i].transpose()
                * 1 / np.sqrt(self._eigenvalues[i])
            )

        # We first need to get the embeddings of the other points
        other_points = self._dataset[
            ~self._dataset.index.isin(self._landmarks.index)
        ]
        other_embeddings = np.vstack(
            other_points['embeddings'].apply(np.array)
        )

        # We compute for each point the distance to the landmarks
        distance_to_landmarks = self._distance_metric_func(
            other_embeddings, self._landmark_embeddings
        )

        # Going through each point, we compute its position
        # by -1/2 * L_sharp * (distance_to_landmarks - mean_distance)
        positions = np.zeros((len(other_points), self._dimension))
        for i in range(len(other_points)):
            position = -1/2 * np.dot(
                L_sharp, distance_to_landmarks[i] - mean_distance
            )
            positions[i, :] = position

        # Append the position of the other points to the dataset
        self._no_landmark_points = other_points.assign(
            position=positions.tolist(), landmark=False
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            'heuristic': self._heuristic,
            'distance_metric': self._distance_metric,
            'num_landmarks': self._num_landmarks,
            'landmarks_selected': self.landmarks_selected,
            'landmarks_reduced': self.landmarks_reduced,
            'points_calculated': self.points_calculated
        }


if __name__ == '__main__':
    import pickle

    with open("./volumes/data/imdb_embeddings.pkl", "rb") as file:
        dataset = pickle.load(file)

    dataset_length = len(dataset)
    print(f"Dataset Length: {dataset_length}")
    print()

    lmds = Lmds(
        heuristic='random',
        distance_metric='euclidean',
        num_landmarks=10,
        dataset=dataset
    )

    lmds.select_landmarks()
    lmds.reduce_landmarks()
    print("Landmarks:")
    print(lmds.landmarks)
    print()

    lmds.calculate()
    print("All points:")
    print(lmds.all_points)
