import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/eofm/12_recover_exports_serial.py"
SPEC = importlib.util.spec_from_file_location("serial_recovery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response:
    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


def setup_recovery(tmp_path):
    recovery = MODULE.Recovery.__new__(MODULE.Recovery)
    recovery.prefix = "eofm_galileo_patch3_2022"
    recovery.manifest = [None] * 13429
    recovery.state_path = tmp_path / "state.json"
    name = recovery.name(19)
    local = tmp_path / name
    backup = tmp_path / "backup.csv"
    local.write_bytes(b"verified CSV fixture\n")
    backup.write_bytes(local.read_bytes())
    sha, md5, size = MODULE.digests(local)
    record = {"chunk": 19, "file": name, "local_path": str(local), "backup_path": str(backup),
              "sha256": sha, "md5": md5, "bytes": size, "cloud": {"id": "exact-temporary-file"},
              "cloud_state": "recoverable_trash"}
    recovery.state = {"chunks": {"19": record}}
    recovery.validate = lambda chunk, path: None
    calls = []

    def api(method, path, **kwargs):
        calls.append((method, path))
        if path == "/about":
            return Response({"storageQuota": {"limit": "25000000", "usage": "1"}})
        return Response({"id": "exact-temporary-file", "name": name, "size": size,
                         "md5Checksum": md5, "trashed": True})
    recovery.api = api
    return recovery, record, calls


def test_backup_mismatch_prevents_all_cloud_access(tmp_path):
    recovery, record, calls = setup_recovery(tmp_path)
    Path(record["backup_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Backup/integrity"):
        recovery.trash(record)
    assert calls == []


def test_same_path_is_not_independent_backup(tmp_path):
    recovery, record, calls = setup_recovery(tmp_path)
    record["backup_path"] = record["local_path"]
    with pytest.raises(ValueError, match="Independent"):
        recovery.trash(record)
    assert calls == []


def test_wrong_cloud_identity_prevents_deletion(tmp_path):
    recovery, record, calls = setup_recovery(tmp_path)
    recovery.api = lambda *a, **k: Response({"name": "original.tif", "size": record["bytes"],
                                           "md5Checksum": record["md5"], "trashed": True})
    with pytest.raises(ValueError, match="identity changed"):
        recovery.trash(record)


def test_low_quota_only_purges_exact_validated_temporary_file(tmp_path):
    recovery, record, calls = setup_recovery(tmp_path)
    recovery.ensure_space()
    assert calls == [("GET", "/about"), ("GET", "/files/exact-temporary-file"),
                     ("DELETE", "/files/exact-temporary-file")]
    assert record["cloud_state"] == "permanently_removed_for_quota"
    assert Path(record["local_path"]).exists()
    assert Path(record["backup_path"]).exists()


def test_sufficient_quota_does_not_delete_any_file(tmp_path):
    recovery, record, calls = setup_recovery(tmp_path)
    recovery.api = lambda *a, **k: Response({"storageQuota": {"limit": "100000000", "usage": "0"}})
    recovery.ensure_space()
    assert record["cloud_state"] == "recoverable_trash"


@pytest.mark.parametrize("error_type", [OSError, MODULE.httplib2.ServerNotFoundError])
def test_transient_ee_read_failure_retries_without_mutation(tmp_path, monkeypatch, error_type):
    recovery, _, cloud_calls = setup_recovery(tmp_path)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    attempts = []

    def read_status():
        attempts.append(1)
        if len(attempts) == 1:
            raise error_type("temporary DNS failure")
        return {"state": "COMPLETED"}

    assert recovery.read_ee(read_status) == {"state": "COMPLETED"}
    assert len(attempts) == 2
    assert cloud_calls == []
