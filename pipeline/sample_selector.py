from __future__ import annotations

import random
from pathlib import Path
from typing import List

from utils.io_utils import list_image_files, read_json, write_json


def load_or_create_manifest(
    *,
    input_dir: Path,
    manifest_path: Path,
    sample_size: int,
    sample_seed: int,
) -> List[Path]:
    if manifest_path.exists():
        payload = read_json(manifest_path)
        items = payload.get("items", [])
        paths = [input_dir / item for item in items]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Manifest contains missing files under {input_dir}: {missing}"
            )
        return paths

    image_paths = list_image_files(input_dir)
    if sample_size > len(image_paths):
        raise ValueError(
            f"Requested sample size {sample_size} exceeds available images {len(image_paths)}."
        )

    rng = random.Random(sample_seed)
    selected = sorted(rng.sample(image_paths, sample_size), key=lambda path: path.name)
    write_json(
        manifest_path,
        {
            "input_dir": str(input_dir),
            "sample_seed": sample_seed,
            "sample_size": sample_size,
            "items": [path.name for path in selected],
        },
    )
    return selected
