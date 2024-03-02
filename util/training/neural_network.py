import torch
import torch.nn as nn


class OneLayerModel(nn.Module):
    # A basic one layer neural network
    # Since we are allowing at max 30 landmarks the top triangle (minus the diagonal) of the matrix will have 29*30/2 = 435 elements
    def __init__(self, in_features=435, param=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, param)
        self.fc2 = nn.Linear(param, in_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TwoLayerModel(nn.Module):
    # A basic 2 layer neural network
    # Since we are allowing at max 30 landmarks the top triangle (minus the diagonal) of the matrix will have 29*30/2 = 435 elements
    def __init__(self, in_features=435, param_list=[32, 16]):
        super().__init__()
        self.fc1 = nn.Linear(in_features, param_list[0])
        self.fc2 = nn.Linear(param_list[0], param_list[1])
        self.fc3 = nn.Linear(param_list[1], in_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
