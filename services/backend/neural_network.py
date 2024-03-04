import torch
import torch.nn as nn


class OneLayerModel(nn.Module):
    # A basic one layer neural network
    # Since we are allowing at max 30 landmarks the top triangle (minus the diagonal) of the matrix will have 29*30/2 = 435 elements
    def __init__(
        self, in_features=435, param=32, inner_activation="relu", end_activation="relu"
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, param)
        self.fc2 = nn.Linear(param, in_features)
        self.inner_activation = make_activation_func(inner_activation)
        self.end_activation = make_activation_func(end_activation)

    def forward(self, x):
        x = self.inner_activation(self.fc1(x))
        x = self.end_activation(self.fc2(x))
        return x


class TwoLayerModel(nn.Module):
    # A basic 2 layer neural network
    # Since we are allowing at max 30 landmarks the top triangle (minus the diagonal) of the matrix will have 29*30/2 = 435 elements
    def __init__(
        self,
        in_features=435,
        param1 = 32,
        param2 = 64,
        inner_activation="relu",
        end_activation="relu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, param1)
        self.fc2 = nn.Linear(param1, param2)
        self.fc3 = nn.Linear(param2, in_features)
        self.inner_activation = make_activation_func(inner_activation)
        self.end_activation = make_activation_func(end_activation)

    def forward(self, x):
        x = self.inner_activation(self.fc1(x))
        x = self.inner_activation(self.fc2(x))
        x = self.end_activation(x)
        return x


def make_activation_func(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {activation}")