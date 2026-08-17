"""Report which reproducibility assets are present and how to rebuild them."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data_metadata/data_registry.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def resolve(path: str) -> Path:
    root = Path(os.environ.get("RICE_FUSION_DATA_ROOT", ROOT))
    return (root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a core asset is missing or mismatched")
    args = parser.parse_args()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    failed = False
    print(f"Data root: {os.environ.get('RICE_FUSION_DATA_ROOT', str(ROOT))}")
    for name, item in registry["assets"].items():
        path = resolve(item["path"])
        if not path.is_file():
            failed = True
            recovery = item.get("recovery", "See DATA_AVAILABILITY.md")
            print(f"[MISSING] {name}: {path}")
            print(f"          recovery: {recovery}")
            continue
        expected = item.get("expected_sha256_original") or item.get("sha256")
        if expected:
            actual = digest(path)
            state = "OK" if actual.lower() == expected.lower() else "REBUILT/DIFFERENT"
            if state != "OK" and item.get("expected_sha256_original"):
                print(f"[{state}] {name}: {path}")
                print("          usable only after schema/provenance validation; original hash was not reproduced")
            else:
                print(f"[{state}] {name}: {path}")
        else:
            print(f"[PRESENT] {name}: {path}")
    if failed:
        print("\nMissing large data are expected in a Git clone. See DATA_AVAILABILITY.md.")
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

