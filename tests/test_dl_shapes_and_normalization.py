import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is an optional dependency for DL baselines")

from src.models import MLPClassifier, TemporalCNNClassifier
from src.training.run_dl_baselines import standardize_train_only


def test_model_forward_shapes():
    assert MLPClassifier(92)(torch.randn(4, 92)).shape == (4, 2)
    model = TemporalCNNClassifier(use_s1=True, use_s2=True)
    assert model(torch.randn(4, 3, 23), torch.randn(4, 1, 23)).shape == (4, 2)


def test_normalization_uses_train_only():
    x = np.array([[0.0], [2.0], [1000.0]], dtype="float32")
    scaled, mean, std = standardize_train_only(x, np.array([0, 1]))
    assert mean[0] == 1.0
    assert std[0] == 1.0
    assert scaled[2, 0] == 999.0
