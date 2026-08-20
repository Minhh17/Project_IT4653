#!/usr/bin/env python3
"""Pack small run evidence for handoff between Kaggle accounts."""

import argparse
import hashlib
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-experiment", help="Export one experiment id")
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Use only after final selection; archives become large.",
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit("Archive exists; choose a new output path")

    source = arguments.runs_dir
    if arguments.only_experiment:
        source = source / arguments.only_experiment
    files = []
    for path in sorted(source.glob("**/*")):
        if not path.is_file():
            continue
        if not arguments.include_checkpoints and path.suffix in {".pt", ".pth", ".ckpt"}:
            continue
        files.append(path)
    if not files:
        raise SystemExit("No run artifacts found in {}".format(source))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arguments.output, "w:gz") as archive:
        for path in files:
            # The archive starts at experiment_id/, ready to merge below runs/.
            archive.add(path, arcname=str(path.relative_to(arguments.runs_dir)), recursive=False)

    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    checksum_path = arguments.output.with_suffix(arguments.output.suffix + ".sha256")
    checksum_path.write_text(digest + "  " + arguments.output.name + "\n", encoding="utf-8")
    print("Packed {} files into {}".format(len(files), arguments.output))
    print("SHA256: {}".format(digest))


if __name__ == "__main__":
    main()
