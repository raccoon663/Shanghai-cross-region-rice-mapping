"""Download the Shanghai 2022 official rice product from a temporary FTP note.

Credentials are parsed at runtime and are never persisted in this repository.
"""

from __future__ import annotations

import argparse
import ftplib
import re
from pathlib import Path, PurePosixPath


def read_note(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("FTP note is not valid UTF-8/GB18030/GBK text")


def parse_credentials(text: str) -> tuple[str, int, str, str]:
    hosts = re.findall(r"ftp://([0-9.]+)", text)
    port = re.search(r"端口[：:]\s*(\d+)", text)
    user = re.search(r"用户(?:名)?[：:]\s*([^\r\n]+)", text)
    password = re.search(r"密码[：:]\s*([^\r\n]+)", text)
    if not hosts or not port or not user or not password:
        raise ValueError("The FTP note does not contain all required connection fields")
    public_hosts = [host for host in hosts if not host.startswith(("10.", "192.168.", "172.16."))]
    return (public_hosts or hosts)[0], int(port.group(1)), user.group(1).strip(), password.group(1).strip()


def walk(ftp: ftplib.FTP, directory: str = "/"):
    try:
        entries = list(ftp.mlsd(directory))
    except ftplib.error_perm:
        entries = []
    if not entries:
        # Some order-download FTP servers accept MLSD but return an empty list.
        # NLST is older but more broadly supported.
        current = ftp.pwd()
        try:
            ftp.cwd(directory)
            names = ftp.nlst()
        finally:
            ftp.cwd(current)
        entries = [(PurePosixPath(name).name, {"type": "unknown"}) for name in names]
    for name, facts in entries:
        if name in (".", ".."):
            continue
        path = str(PurePosixPath(directory) / name)
        if facts.get("type") == "dir":
            yield from walk(ftp, path)
        elif facts.get("type") == "file":
            yield path
        else:
            current = ftp.pwd()
            try:
                ftp.cwd(path)
            except ftplib.error_perm:
                yield path
            else:
                ftp.cwd(current)
                yield from walk(ftp, path)


def matches(path: str) -> bool:
    normalized = path.lower()
    province = "shanghai" in normalized or "上海" in path
    return province and "2022" in normalized and normalized.endswith((".tif", ".tiff"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/official_reference"))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    host, port, user, password = parse_credentials(read_note(args.note))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ftplib.FTP() as ftp:
        try:
            ftp.connect(host, port, timeout=args.timeout)
            ftp.login(user, password)
        except OSError as exc:
            raise ConnectionError(
                f"Could not connect to the official FTP endpoint {host}:{port}. "
                "The server may be offline or blocked by the current network."
            ) from exc
        remote_files = sorted(walk(ftp))
        manifest = args.output_dir / "ftp_manifest.txt"
        manifest.write_text("\n".join(remote_files), encoding="utf-8")
        candidates = [path for path in remote_files if matches(path)]
        if not candidates:
            # This dataset is commonly delivered as one archive per year.
            candidates = [
                path for path in remote_files
                if "2022" in path.lower()
                and "rice" in path.lower()
                and path.lower().endswith(".zip")
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No Shanghai GeoTIFF or 2022 rice archive was found. "
                    f"The full server file list was saved to: {manifest.resolve()}"
                )
        if len(candidates) > 1:
            candidate_file = args.output_dir / "ftp_candidates.txt"
            candidate_file.write_text("\n".join(candidates), encoding="utf-8")
            raise RuntimeError(
                f"Multiple matching files found; candidates saved to: {candidate_file.resolve()}"
            )
        remote = candidates[0]
        destination = args.output_dir / PurePosixPath(remote).name
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as stream:
            ftp.retrbinary(f"RETR {remote}", stream.write, blocksize=1024 * 1024)
        temporary.replace(destination)
        print(f"Downloaded: {destination.resolve()}")


if __name__ == "__main__":
    main()
