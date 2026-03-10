from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(path: Path) -> List[Path]:
    files = [
        p
        for p in sorted(path.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return files


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def safe_stem(path: Path) -> str:
    return path.stem.strip()
