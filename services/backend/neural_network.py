import torch.nn as nn

# The file needs to reside in the docker container for inference


class OneLayerModel(nn.Module):
    # A basic one layer neural network
    # Since we are allowing at max 30 landmarks the top triangle
    # (minus the diagonal) of the matrix will have 29 * 30 / 2 = 435 elements
    def __init__(
        self,
        in_features=435,
        param=32,
        inner_activation="relu",
        end_activation="relu",
        dropout_prob=0.0,
    ):
        super().__init__()
        self.dropout_in = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(in_features, param)
        self.dropout_inner = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(param, in_features)
        self.inner_activation = make_activation_func(inner_activation)
        self.end_activation = make_activation_func(end_activation)

    def forward(self, x):
        x = self.dropout_in(x)
        x = self.inner_activation(self.fc1(x))
        x = self.dropout_inner(x)
        x = self.end_activation(self.fc2(x))
        return x


class TwoLayerModel(nn.Module):
    # A basic 2 layer neural network
    # Since we are allowing at max 30 landmarks the top triangle
    # (minus the diagonal) of the matrix will have 29*30/2 = 435 elements
    # It was not used in the end experiments, since the one layer model was sufficient
    def __init__(
        self,
        in_features=435,
        param1=32,
        param2=64,
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
        x = self.end_activation(self.fc3(x))
        return x


class TrivialModel(nn.Module):
    # A trivial model that does nothing, used for testing
    def __init__(self):
        super().__init__()

    def forward(self, x):
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
