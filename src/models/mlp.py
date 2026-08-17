import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 96, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

