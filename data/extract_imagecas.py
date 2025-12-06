"""
Extract the imagecas dataset from a top-level zip and its split archives.

Workflow:
- imagecas.zip contains the split archives (e.g., 1-200.change2zip + .z01/.z02...).
- We unzip imagecas.zip to a destination directory.
- We rename each *.change2zip to *.zip so the system unzip tool recognizes it.
- We call `unzip` on each renamed archive; unzip will consume the matching .z01/.z02 parts.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def unzip_main_archive(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"main archive not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"Extracted {zip_path} -> {extract_dir}")


def rename_split_archives(root: Path) -> list[Path]:
    renamed: list[Path] = []
    for part in sorted(root.glob("*.change2zip")):
        target = part.with_suffix(".zip")
        if target.exists():
            print(f"Skip rename (already exists): {target}")
            continue
        part.rename(target)
        renamed.append(target)
        print(f"Renamed {part.name} -> {target.name}")
    return renamed


def ensure_unzip_available() -> None:
    if shutil.which("unzip") is None:
        raise RuntimeError("`unzip` command not found; install it and re-run.")


def ensure_zip_available() -> None:
    if shutil.which("zip") is None:
        raise RuntimeError("`zip` command not found; needed for zip -FF repair.")


def repair_archive(archive: Path, root: Path) -> Path:
    """
    Attempt to repair a split archive using `zip -FF`.
    Returns path to the repaired archive.
    """
    ensure_zip_available()
    fixed = archive.with_name(f"{archive.stem}_fixed.zip")
    cmd = ["zip", "-FF", archive.name, "--out", fixed.name]
    print(f"Repairing split archive: {' '.join(cmd)} (cwd={root})")
    subprocess.run(cmd, cwd=root, check=True)
    return fixed


def extract_split_archives(root: Path, archives: list[Path]) -> None:
    ensure_unzip_available()
    for archive in archives:
        # Verify companion parts exist (z01, z02, ...).
        stem = archive.with_suffix("")
        parts = sorted(root.glob(f"{stem.name}.z*"))
        if not parts:
            print(f"Warning: no .z* parts found for {archive.name}; attempting unzip anyway.")
        cmd = ["unzip", "-o", archive.name]
        print(f"Running: {' '.join(cmd)} (cwd={root})")
        try:
            subprocess.run(cmd, cwd=root, check=True)
        except subprocess.CalledProcessError:
            print(f"Unzip failed for {archive.name}; attempting repair with zip -FF.")
            fixed = repair_archive(archive, root)
            repair_cmd = ["unzip", "-o", fixed.name]
            print(f"Running: {' '.join(repair_cmd)} (cwd={root})")
            subprocess.run(repair_cmd, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract imagecas split zip archives.")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("/Users/arvind/veevohealth/imagecas-dataset/imagecas-dataset/imagecas.zip"),
        help="Path to imagecas.zip containing the split parts.",
    )
    parser.add_argument(
        "--dest",
        dest="dest_dir",
        type=Path,
        default=Path("/Users/arvind/veevohealth/imagecas-dataset/imagecas-dataset"),
        help="Directory to extract into; should contain/receive the split parts.",
    )
    args = parser.parse_args()

    zip_path: Path = args.zip_path.expanduser().resolve()
    dest_dir: Path = args.dest_dir.expanduser().resolve()

    try:
        unzip_main_archive(zip_path, dest_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed extracting main archive: {exc}", file=sys.stderr)
        sys.exit(1)

    renamed = rename_split_archives(dest_dir)
    if not renamed:
        print("No *.change2zip files found; nothing to extract.")
        return

    try:
        extract_split_archives(dest_dir, renamed)
    except subprocess.CalledProcessError as exc:
        print(f"Unzip failed: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

