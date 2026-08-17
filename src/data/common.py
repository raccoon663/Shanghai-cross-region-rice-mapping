from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config, path


def resolve_project_path(value: str | Path, config_path: Path) -> Path:
    value = os.path.expandvars(str(value))
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


def resolve_data_path(value: str | Path, config_path: Path) -> Path:
    """Resolve a data path relative to RICE_FUSION_DATA_ROOT or the repository.

    The environment variable lets users keep large public-data exports outside
    Git without editing committed configuration files.
    """
    value = os.path.expandvars(str(value))
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    data_root = os.environ.get("RICE_FUSION_DATA_ROOT")
    if data_root:
        return (Path(data_root).expanduser() / path).resolve()
    return resolve_project_path(path, config_path)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            "Run `python scripts/check_data.py` for recovery instructions, or "
            "set RICE_FUSION_DATA_ROOT to a directory containing the paths "
            "listed in configs/data.yaml."
        )
    return path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gee_points(frame: pd.DataFrame) -> pd.DataFrame:
    if ".geo" not in frame:
        raise ValueError("Source table has no '.geo' field containing GEE point geometry")
    coords = frame[".geo"].map(lambda value: json.loads(value)["coordinates"])
    result = frame.copy()
    result["longitude"] = coords.map(lambda value: float(value[0]))
    result["latitude"] = coords.map(lambda value: float(value[1]))
    return result


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    def ordered(prefix: str) -> list[str]:
        return sorted(
            [name for name in columns if name.startswith(prefix)],
            key=lambda name: int(name.rsplit("t", 1)[1]),
        )

    s2 = ordered("s2_ndvi_t")
    s1 = ordered("s1_vv_t") + ordered("s1_vh_t") + ordered("s1_rvi_t")
    groups = {"s2_only": s2, "s1_only": s1, "s1_s2_fusion": s2 + s1}
    expected = {"s2_only": 23, "s1_only": 69, "s1_s2_fusion": 92}
    for name, features in groups.items():
        if len(features) != expected[name] or len(features) != len(set(features)):
            raise ValueError(
                f"Unexpected {name} feature schema: found {len(features)}, "
                f"expected {expected[name]} unique columns"
            )
    return groups
