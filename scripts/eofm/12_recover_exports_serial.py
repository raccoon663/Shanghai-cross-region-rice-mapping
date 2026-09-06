"""Recover frozen EOFM exports serially, with verified backups before cloud cleanup.

The image graph reuses 02_submit_presto_point_exports._stack. Galileo geometry
is the direct Python equivalent of the frozen 06 Code Editor generator.
No embedding extraction, labels, model fitting, or downstream evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import httplib2
import pandas as pd
import requests
import yaml
from google.auth.transport.requests import AuthorizedSession, Request

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_HASH = "04c4e7454ead6cd415f5d761593d9e88b8041dd6611f630064fca798ccf6421b"
GRID_HASH = "133591caab146edd844a6f395828b9eda153fe810173fe7c44d569e92c68497a"
DRIVE = "https://www.googleapis.com/drive/v3"


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = module("02_submit_presto_point_exports.py", "frozen_point_export")
VALIDATE = module("11_validate_recovery_chunks.py", "frozen_chunk_validation")
PATCH = module("09_prepare_galileo_smoke_patches.py", "frozen_patch_schema")
MONTHLY = module("08_prepare_presto_monthly_inputs.py", "frozen_monthly_schema")


def log(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digests(path):
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return sha.hexdigest(), md5.hexdigest(), size


def patch_collection(ee, centers, projection):
    def expand(feature):
        xy = ee.List(feature.geometry().transform(projection, 1).coordinates())
        pixels = []
        for row in range(3):
            for col in range(3):
                x = ee.Number(xy.get(0)).add(ee.Number(col - 1).multiply(10))
                y = ee.Number(xy.get(1)).add(ee.Number(1 - row).multiply(10))
                pixels.append(ee.Feature(ee.Geometry.Point([x, y], projection), {
                    "manifest_row": feature.get("manifest_row"),
                    "chunk_id": feature.get("chunk_id"), "region": feature.get("region"),
                    "patch_row": row, "patch_col": col,
                }))
        return ee.FeatureCollection(pixels)
    return ee.FeatureCollection(centers.map(expand).flatten())


def export_graph(ee, config, grid, manifest, chunk, patch):
    start, stop = chunk * 500, min((chunk + 1) * 500, len(manifest))
    centers = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row.longitude), float(row.latitude)]), {
            "manifest_row": int(row.Index), "chunk_id": chunk,
            "region": "jiangxi" if row.Index < 1429 else "shanghai",
        }) for row in manifest.iloc[start:stop].itertuples()
    ])
    transform = list(config["earth_engine"]["sample_crs_transform"])
    if patch:
        transform[0], transform[4] = 10, -10
    projection = ee.Projection(config["earth_engine"]["sample_crs"], transform)
    points = patch_collection(ee, centers, projection) if patch else centers
    # Match frozen Code Editor filterBounds(centers), not geometry().bounds().
    image = BASE._stack(ee, config, grid, centers)
    properties = ["manifest_row", "region"] + (["patch_row", "patch_col"] if patch else [])
    sampled = image.sampleRegions(collection=points, properties=properties,
                                  projection=projection, geometries=False, tileScale=4)
    return centers, points, image, sampled


class Recovery:
    def __init__(self, args):
        self.args = args
        self.patch = args.representation == "galileo"
        self.config = yaml.safe_load((ROOT / ("configs/eofm_input_reconstruction.yaml" if self.patch
                                             else "configs/eofm_presto_monthly.yaml")).read_text())
        self.manifest_path = ROOT / self.config["sample_manifest"]
        self.grid_path = ROOT / self.config["temporal_grid_manifest"]
        if BASE.sha256(self.manifest_path) != SAMPLE_HASH:
            raise ValueError("Frozen sample checksum changed")
        self.manifest = BASE.validate_manifest(self.manifest_path)
        self.grid = json.loads(self.grid_path.read_text())
        if not args.project or not args.project.strip():
            raise ValueError("An explicit Earth Engine project is required")
        self.project = args.project.strip()
        if self.patch:
            if BASE.sha256(self.grid_path) != GRID_HASH:
                raise ValueError("Frozen Galileo grid checksum changed")
            galileo = self.config["galileo_input"]
            if (galileo["patch_pixels"], galileo["sample_resolution_m"], galileo["encoder_patch_size"]) != (3, 10, 1):
                raise ValueError("Frozen Galileo design changed")
        else:
            if len(self.grid["windows"]) != 12 or self.grid["month_argument_zero_based"] != 0:
                raise ValueError("Frozen monthly design changed")
        tags = [w["tag"] for w in self.grid["windows"]]
        bands = self.config["earth_engine"]["sentinel_2_bands"]
        self.columns = (PATCH.expected_columns if self.patch else MONTHLY.columns_for)(bands, tags)
        self.prefix = "eofm_galileo_patch3_2022" if self.patch else "eofm_presto_monthly_2022"
        self.folder = "rice_eofm_galileo_patch3_2022" if self.patch else "rice_eofm_presto_monthly_2022"
        self.local_dir = ROOT / ("outputs/eofm/galileo_full_downloaded" if self.patch
                                 else "outputs/eofm/ee/presto_monthly_downloaded")
        self.report_path = ROOT / f"results/manifests/eofm_{args.representation}_recovery_validation.json"
        self.state_path = ROOT / f"outputs/eofm/ee/{args.representation}_serial_recovery.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"chunks": {}, "attempts": []}
        previous_project = self.state.get("project")
        if previous_project is None and self.state["attempts"]:
            previous_project = self.config["earth_engine"]["project"]
        if previous_project and previous_project != self.project:
            raise ValueError("Recovery ledger belongs to a different Earth Engine project")
        self.state["project"] = self.project
        self.local_dir.mkdir(parents=True, exist_ok=True)
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        self.ee = None

    def connect(self):
        import ee
        import httplib2
        import socks
        self.ee = ee
        transport = requests.Session()
        transport.trust_env = False
        proxy_info = None
        if self.args.proxy_port:
            proxy = f"http://127.0.0.1:{self.args.proxy_port}"
            transport.proxies.update({"https": proxy})
            proxy_info = httplib2.ProxyInfo(socks.PROXY_TYPE_HTTP, "127.0.0.1", self.args.proxy_port)
        self.drive = AuthorizedSession(ee.data.get_persistent_credentials(), auth_request=Request(session=transport))
        self.drive.trust_env = False
        self.drive.proxies.update(transport.proxies)
        ee.Initialize(project=self.project, http_transport=httplib2.Http(timeout=60,
                      proxy_info=proxy_info))

    def name(self, chunk):
        return f"{self.prefix}_c{chunk:03d}_{chunk*500:05d}_{min((chunk+1)*500,len(self.manifest)):05d}.csv"

    def api(self, method, path, **kwargs):
        for attempt in range(3):
            try:
                response = self.drive.request(method, DRIVE + path, timeout=(20, 60), **kwargs)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def read_ee(self, operation):
        """Retry transport failures on read-only EE calls without resubmitting exports."""
        for attempt in range(5):
            try:
                return operation()
            except (OSError, http.client.HTTPException, httplib2.ServerNotFoundError) as error:
                log("ee_read_network_retry", attempt=attempt + 1, error=type(error).__name__)
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))

    def cloud_file(self, name):
        files = self.api("GET", "/files", params={"q": f"name = '{name}'", "pageSize": 100,
                     "fields": "files(id,name,size,md5Checksum,trashed)"}).json()["files"]
        if len(files) > 1:
            raise ValueError(f"Ambiguous duplicate cloud filename: {name}")
        return files[0] if files else None

    def validate(self, chunk, path):
        return VALIDATE.validate_frame(pd.read_csv(path), self.columns, self.manifest,
                    chunk * 500, min((chunk + 1) * 500, len(self.manifest)), self.patch)

    def record_local(self, chunk, metadata):
        path = self.local_dir / self.name(chunk)
        backup = self.args.backup_dir / path.name
        checks = self.validate(chunk, path)
        sha, md5, size = digests(path)
        if not backup.exists():
            # Exclusive creation prevents overwriting any earlier backup.
            with path.open("rb") as source, backup.open("xb") as target:
                shutil.copyfileobj(source, target)
        if digests(backup) != (sha, md5, size):
            raise ValueError("Backup checksum mismatch: stop without cloud deletion")
        if metadata and (metadata["name"] != path.name or int(metadata["size"]) != size or metadata["md5Checksum"] != md5):
            raise ValueError("Cloud/local integrity mismatch: stop without cloud deletion")
        previous = self.state["chunks"].get(str(chunk))
        if previous and previous["sha256"] != sha:
            raise ValueError("Previously validated checksum changed")
        record = {"chunk": chunk, "file": path.name, "sha256": sha, "md5": md5, "bytes": size,
                  "local_path": str(path), "backup_path": str(backup), "cloud": metadata,
                  "cloud_state": "present" if metadata else "absent", **checks}
        self.state["chunks"][str(chunk)] = record
        save(self.state_path, self.state)
        self.update_report()
        log("local_validated", chunk=chunk, rows=checks["rows"], sha256=sha)
        return record

    def update_report(self):
        # Reuse the independently tested complete/partial ledger validator.
        import subprocess
        import sys
        command = [sys.executable, str(Path(__file__).with_name("11_validate_recovery_chunks.py")),
                   "--representation", self.args.representation, "--chunk-dir", str(self.local_dir),
                   "--report", str(self.report_path)]
        if self.patch:
            command += ["--chunk-dir", str(ROOT / "outputs/eofm/galileo_smoke_downloaded")]
        subprocess.run(command, cwd=ROOT, check=True)

    def verify_cleanup(self, record):
        chunk = record["chunk"]
        if not 0 <= chunk < 27 or record["file"] != self.name(chunk):
            raise ValueError("Cleanup target is outside the frozen temporary CSV scope")
        path, backup = Path(record["local_path"]), Path(record["backup_path"])
        if path.resolve() == backup.resolve():
            raise ValueError("Independent retained backup is required")
        expected = (record["sha256"], record["md5"], record["bytes"])
        if digests(path) != expected or digests(backup) != expected:
            raise ValueError("Backup/integrity failure: do not delete cloud data")
        self.validate(chunk, path)
        current = self.api("GET", "/files/" + record["cloud"]["id"],
                           params={"fields": "id,name,size,md5Checksum,trashed"}).json()
        if (current.get("id"), current["name"], int(current["size"]), current["md5Checksum"]) != (record["cloud"]["id"], record["file"], record["bytes"], record["md5"]):
            raise ValueError("Cloud deletion identity changed")
        return current

    def trash(self, record):
        if not self.args.cleanup_cloud:
            return
        if not record["cloud"]:
            return
        current = self.verify_cleanup(record)
        if not current["trashed"]:
            self.api("PATCH", "/files/" + current["id"], json={"trashed": True})
        record["cloud_state"] = "recoverable_trash"
        save(self.state_path, self.state)
        log("cloud_trashed", chunk=record["chunk"], local_backups_retained=True)

    def ensure_space(self):
        quota = self.api("GET", "/about", params={"fields": "storageQuota"}).json()["storageQuota"]
        free = int(quota["limit"]) - int(quota["usage"])
        # A conservative operational margin; does not alter sample/chunk design.
        if free >= 25_000_000:
            return
        if not (self.args.cleanup_cloud and self.args.purge_cloud_for_quota):
            raise RuntimeError("Insufficient quota; cloud files retained. Permanent cleanup requires --cleanup-cloud --purge-cloud-for-quota")
        for record in self.state["chunks"].values():
            if record["cloud_state"] != "recoverable_trash":
                continue
            current = self.verify_cleanup(record)
            if not current["trashed"]:
                raise ValueError("Cleanup state changed")
            self.api("DELETE", "/files/" + current["id"])
            record["cloud_state"] = "permanently_removed_for_quota"
            save(self.state_path, self.state)
            log("cloud_purged_for_space", chunk=record["chunk"], bytes=record["bytes"], local_backups_retained=True)
            free += record["bytes"]
            if free >= 25_000_000:
                return
        raise RuntimeError("Insufficient quota and no further verified temporary CSV available")

    def download(self, chunk, metadata):
        output = self.local_dir / self.name(chunk)
        for attempt in range(3):
            partial = output.with_suffix(f".csv.{uuid.uuid4().hex}.part")
            try:
                response = self.api("GET", "/files/" + metadata["id"], params={"alt": "media"}, stream=True)
                with partial.open("xb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                sha, md5, size = digests(partial)
                if size != int(metadata["size"]) or md5 != metadata["md5Checksum"]:
                    raise ValueError("Downloaded CSV checksum/size failed; preserve partial and stop")
                self.validate(chunk, partial)
                if output.exists():
                    raise ValueError("Concurrent local destination appeared; do not overwrite")
                partial.rename(output)
                return self.record_local(chunk, metadata)
            except (requests.ConnectionError, requests.Timeout):
                log("download_network_retry", chunk=chunk, attempt=attempt + 1)
                if attempt == 2:
                    raise
                time.sleep(5)

    def obtain(self, chunk):
        name = self.name(chunk)
        cloud = self.cloud_file(name)
        if (self.local_dir / name).exists():
            return self.record_local(chunk, cloud)
        for retry in range(3):
            tasks = [t for t in self.read_ee(self.ee.batch.Task.list) if t.config.get("description") == name[:-4]]
            active = [t for t in tasks if t.state in ("READY", "RUNNING", "CANCEL_REQUESTED")]
            if len(active) > 1:
                raise RuntimeError("Multiple active exports for one frozen chunk")
            if cloud:
                return self.download(chunk, cloud)
            if active:
                task = active[0]
                if task.state == "CANCEL_REQUESTED":
                    raise RuntimeError("Existing cancellation must finish before recovery")
            else:
                if any(t.state == "COMPLETED" for t in tasks):
                    raise RuntimeError("Completed task has no visible CSV/local copy; investigate before resubmit")
                self.ensure_space()
                centers, points, image, sampled = export_graph(self.ee, self.config, self.grid, self.manifest, chunk, self.patch)
                task = self.ee.batch.Export.table.toDrive(collection=sampled, description=name[:-4], folder=self.folder,
                            fileNamePrefix=name[:-4], fileFormat="CSV", selectors=self.columns)
                task.start()
                self.state["attempts"].append({"chunk": chunk, "task_id": task.id, "state": "SUBMITTED"})
                save(self.state_path, self.state)
                log("submitted", chunk=chunk, task_id=task.id)
            previous = None
            while True:
                status = self.read_ee(task.status)
                state = status["state"]
                if state != previous:
                    log("task_state", chunk=chunk, task_id=task.id, state=state)
                    previous = state
                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    self.state["attempts"].append({"chunk": chunk, "task_id": task.id,
                          "state": state, "error_message": status.get("error_message")})
                    save(self.state_path, self.state)
                    break
                time.sleep(20)
            if state == "COMPLETED":
                for _ in range(6):
                    cloud = self.cloud_file(name)
                    if cloud:
                        return self.download(chunk, cloud)
                    time.sleep(10)
                raise RuntimeError("Completed export not yet visible; stop without resubmitting")
            log("export_retry", chunk=chunk, attempt=retry + 1, reason=status.get("error_message"))
            if state == "CANCELLED":
                raise RuntimeError("Export cancelled; do not override external cancellation")
        raise RuntimeError("Chunk export failed three times without changing design")

    def run(self):
        self.connect()
        if self.args.check_plan:
            centers, points, image, _ = export_graph(self.ee, self.config, self.grid, self.manifest, self.args.start, self.patch)
            expected = min(500, len(self.manifest) - self.args.start * 500)
            values = self.ee.Dictionary({"centers": centers.size(), "rows": points.size(),
                                         "selected_bands": image.select(self.columns[4 if self.patch else 2:]).bandNames().size()}).getInfo()
            assert values == {"centers": expected, "rows": expected * (9 if self.patch else 1),
                              "selected_bands": len(self.columns) - (4 if self.patch else 2)}, values
            log("graph_contract_passed", **values)
            return
        if self.args.cleanup_previous and self.args.start > 0:
            chunk = self.args.start - 1
            if (self.local_dir / self.name(chunk)).exists():
                self.trash(self.record_local(chunk, self.cloud_file(self.name(chunk))))
        for chunk in range(self.args.start, self.args.stop):
            record = self.obtain(chunk)
            self.trash(record)
        log("requested_range_complete", representation=self.args.representation, start=self.args.start, stop=self.args.stop)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["galileo", "presto_primary"], required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("EARTH_ENGINE_PROJECT"),
                        help="Explicit execution project, or EARTH_ENGINE_PROJECT; must match an existing recovery ledger")
    parser.add_argument("--proxy-port", type=int, default=0, help="Optional local HTTP proxy port; default 0 uses direct access")
    parser.add_argument("--cleanup-cloud", action="store_true", help="Trash only validated temporary CSV exports with retained independent backups")
    parser.add_argument("--purge-cloud-for-quota", action="store_true", help="Allow permanent deletion of verified trashed CSV exports when quota is low; requires --cleanup-cloud")
    parser.add_argument("--cleanup-previous", action="store_true")
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.project or not args.project.strip():
        parser.error("Supply --project or EARTH_ENGINE_PROJECT explicitly")
    if (args.cleanup_previous or args.purge_cloud_for_quota) and not args.cleanup_cloud:
        parser.error("--cleanup-previous and --purge-cloud-for-quota require --cleanup-cloud")
    if not 0 <= args.proxy_port <= 65535:
        parser.error("--proxy-port must be 0 or a valid TCP port")
    if not 0 <= args.start < args.stop <= 27:
        raise ValueError("Chunk range must remain within frozen 0..26")
    args.backup_dir = args.backup_dir.resolve()
    return args


def main():
    args = parse_args()
    recovery = Recovery(args)
    lock = ROOT / "outputs/eofm/ee/serial_export.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        recovery.run()
    finally:
        if lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
