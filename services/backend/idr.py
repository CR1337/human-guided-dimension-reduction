import os
import numpy as np
from typing import Any, List

from inference import Predictor


class InverseDimensionaltyReduction:

    NEURAL_NETWORK_NAMES: List[str] = ["nn_imdb"]
    OTHER_NAMES: List[str] = ["none", "trivial"]
    VALID_NAMES: List[str] = OTHER_NAMES + NEURAL_NETWORK_NAMES

    DOCKER_PATH: str = "/server/models"
    LOCAL_PATH: str = "./volumes/models"

    INSIDE_DOCKER = bool(os.environ.get('INSIDE_DOCKER', False))

    _name: str
    _distance_metric: str
    _is_neural_network: bool

    _model_path: str

    def __init__(self, name: str, distance_metric: str, method: str):
        self._name = name
        self._distance_metric = distance_metric
        self._is_neural_network = name in self.NEURAL_NETWORK_NAMES
        if self._is_neural_network:
            self._name += (
                f"_mds_{distance_metric}" if method == "MDS"
                else f"_tsne_{distance_metric}"
            )

            self._model_path = os.path.join(
                self.DOCKER_PATH if self.INSIDE_DOCKER else self.LOCAL_PATH,
                self._name
            )

    def inference(
        self, distance_matrix: Any, old_delta_n: Any
    ) -> np.ndarray:
        if self._name in self.OTHER_NAMES:
            return self._other_inference(distance_matrix, old_delta_n)
        else:
            return self._neural_network_inference(distance_matrix)

    def _neural_network_inference(self, distance_matrix: Any) -> np.ndarray:
        predictor = Predictor(model_path=self._model_path)
        return predictor.inference(distance_matrix) ** 2

    def _other_inference(
        self, distance_matrix: Any, old_delta_n: Any
    ) -> np.ndarray:
        if self._name == "none":
            return old_delta_n
        elif self._name == "trivial":
            return distance_matrix ** 2
        else:
            raise ValueError(f"Invalid iDR Algorithm: {self._name}")
