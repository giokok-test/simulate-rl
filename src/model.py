import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePolicyNetwork(nn.Module):
    """A simple feed-forward policy network."""

    def __init__(self, input_dim: int = 9, action_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
