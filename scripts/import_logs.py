#!/usr/bin/env python3
"""Safely merge non-conflicting run archives from the three members."""

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe archive member: {}".format(name))
    return Path(*path.parts)


def _verify_checksum(archive_path):
    """Verify export_logs.py's companion checksum when it is available."""
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if not checksum_path.exists():
        print("WARNING: no companion checksum for {}".format(archive_path))
        return
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("Checksum mismatch: {}".format(archive_path))
    print("Checksum OK: {}".format(archive_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    arguments = parser.parse_args()
    arguments.runs_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    identical = 0
    for archive_path in arguments.archives:
        _verify_checksum(archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = _safe_relative_path(member.name)
                target = arguments.runs_dir / relative
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise RuntimeError("Cannot read {} from {}".format(member.name, archive_path))
                content = source_file.read()
                if target.exists():
                    if target.read_bytes() != content:
                        raise SystemExit("Conflict: {} differs across archives".format(target))
                    identical += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                imported += 1
    print("Imported {} files; skipped {} identical files".format(imported, identical))


if __name__ == "__main__":
    main()
