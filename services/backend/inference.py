import torch
import os
import yaml

import numpy as np
from typing import List

from data_loading import process_single_input
import neural_network

class Predictor:
    def __init__(self):
        self.nn, max_landmarks = self.load_nn()
        self.max_landmarks = max_landmarks

    def load_nn(self):
        # Load the model from checkpoints/best
        model_folder = '/server/checkpoints/best'
        weights = torch.load(os.path.join(model_folder, 'model.ckpt'))
        params = yaml.safe_load(open(os.path.join(model_folder, 'params.yml')))
        if params["model_name"] == "OneLayerModel":
            nn = neural_network.OneLayerModel(
                params["max_input_size"],
                params["model_params"],
                params["inner_activation"],
                params["end_activation"],
            )
        elif params["model_name"] == "TwoLayerModel":
            nn = neural_network.TwoLayerModel(
                params["max_input_size"],
                params["model_params"],
                params["inner_activation"],
                params["end_activation"],
            )
        else:
            raise ValueError(f"Unknown model name: {params['model_name']}")
        nn.load_state_dict(weights)
        return nn, params["max_landmarks"]

    def inference(self, distance_matrix):
        # Process the distance matrix
        nn_input = process_single_input(distance_matrix, max_landmarks=self.max_landmarks)
        # Get a distance matrix and call self.model(distance_matrix)
        with torch.no_grad():
            upper_triangle = self.nn.forward(nn_input)
        # Return the result to matrix form
        return self.unprocess(upper_triangle, distance_matrix.shape[0])

    def unprocess(self, result: List[float], num_landmarks: int) -> np.ndarray:
        # We need to construct the distance matrix from the upper triangle
        size = self.max_landmarks
        square = np.zeros((size, size))
        indices = np.triu_indices(size, k=1)
        square[indices] = result
        square.T[indices] = result

        # We need to remove any used padding
        return square[:num_landmarks, :num_landmarks]