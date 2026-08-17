from pathlib import Path

import yaml

from src.data.common import resolve_data_path


def test_public_data_config_has_no_machine_specific_paths(monkeypatch):
    monkeypatch.delenv("RICE_FUSION_DATA_ROOT", raising=False)
    config_path = Path("configs/data.yaml").resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key, value in config["data"].items():
        if value is None:
            continue
        assert not Path(value).is_absolute(), f"{key} must be portable: {value}"


def test_data_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RICE_FUSION_DATA_ROOT", str(tmp_path))
    config_path = Path("configs/data.yaml").resolve()
    assert resolve_data_path("data/raw/example.csv", config_path) == (tmp_path / "data/raw/example.csv").resolve()
