#!/usr/bin/env python3
import shutil
from pathlib import Path

# --- CONFIG: adjust these if needed ---
# Current ImageCAS data root (contains 1-200, 201-400, ...)
SOURCE_ROOT = Path("/Users/arvind/veevohealth/imagecas-dataset/imagecas-dataset")

# Where you want the per-ID folders created
# (change this if you prefer a different location)
DEST_ROOT = Path("/Users/arvind/veevohealth/imagecas-dataset/imagecas-dataset")

# True = copy files (safer), False = move files (originals removed)
COPY_INSTEAD_OF_MOVE = True
# --------------------------------------


def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Go through each range directory: 1-200, 201-400, ...
    for range_dir in sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir()):
        print(f"Processing directory: {range_dir}")
        for f in sorted(range_dir.glob("*.nii.gz")):
            # Expect names like "1.img.nii.gz" or "1.label.nii.gz"
            parts = f.name.split(".")
            if len(parts) != 4:
                # Skip anything that does not match expected pattern
                print(f"  Skipping unexpected file name: {f.name}")
                continue

            case_id, kind, _, _ = parts  # e.g. ["1", "img", "nii", "gz"]
            if kind not in {"img", "label"}:
                print(f"  Skipping unexpected kind in file name: {f.name}")
                continue

            case_dir = DEST_ROOT / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            dest = case_dir / f.name  # keep original file names
            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(f, dest)
                action = "Copied"
            else:
                shutil.move(str(f), dest)
                action = "Moved"

            print(f"  {action} {f} -> {dest}")


if __name__ == "__main__":
    main()