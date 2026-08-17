import torch
from torch import nn


class TemporalBranch(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TemporalCNNClassifier(nn.Module):
    def __init__(self, use_s1: bool, use_s2: bool, hidden_size: int = 96, dropout: float = 0.2):
        super().__init__()
        if not use_s1 and not use_s2:
            raise ValueError("At least one sensor branch is required")
        self.use_s1, self.use_s2 = use_s1, use_s2
        self.s1_branch = TemporalBranch(3) if use_s1 else None
        self.s2_branch = TemporalBranch(1) if use_s2 else None
        branch_size = 32 * int(use_s1) + 32 * int(use_s2)
        self.head = nn.Sequential(
            nn.Linear(branch_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, s1: torch.Tensor | None, s2: torch.Tensor | None) -> torch.Tensor:
        embeddings = []
        if self.use_s1:
            embeddings.append(self.s1_branch(s1))
        if self.use_s2:
            embeddings.append(self.s2_branch(s2))
        return self.head(torch.cat(embeddings, dim=1))

