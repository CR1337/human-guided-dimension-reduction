import torch
import torch.nn as nn

class TwoLayerModel(nn.Module):
    # A basic 2 layer neural network
    # Since we are allowing at max 30 landmarks the distance matrix is max 90 big
    def __init__(self, in_features=90, param_list=[32, 16]):
        super().__init__()
        if len(param_list) != 2:
            raise ValueError("The model_params list should have 2 elements")
        self.fc1 = nn.Linear(in_features, param_list[0])
        self.fc2 = nn.Linear(param_list[0], param_list[1])
        self.fc3 = nn.Linear(param_list[1], in_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x