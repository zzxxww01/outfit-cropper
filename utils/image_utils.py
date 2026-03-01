from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


BBoxXYXY = Tuple[int, int, int, int]


def clamp_bbox_xyxy(bbox_xyxy: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return int(x1), int(y1), int(x2), int(y2)


def expand_bbox_xyxy(
    bbox_xyxy: BBoxXYXY,
    width: int,
    height: int,
    pad_ratio: float = 0.1,
) -> BBoxXYXY:
    x1, y1, x2, y2 = bbox_xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))
    expanded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    return clamp_bbox_xyxy(expanded, width, height)


def bbox_xyxy_to_1000(
    bbox_xyxy: BBoxXYXY,
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    ymin = (y1 / max(1, height)) * 1000.0
    xmin = (x1 / max(1, width)) * 1000.0
    ymax = (y2 / max(1, height)) * 1000.0
    xmax = (x2 / max(1, width)) * 1000.0
    return [round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)]


def mask_to_bbox_xyxy(mask: np.ndarray) -> BBoxXYXY | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def composite_on_white(image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be HxWx3.")
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be HxW.")
    white = np.full_like(image_rgb, 255)
    keep = binary_mask > 0
    white[keep] = image_rgb[keep]
    return white


def crop_rgb(image_rgb: np.ndarray, bbox_xyxy: BBoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return image_rgb[y1:y2, x1:x2]


def image_to_jpg_base64(image: Image.Image, quality: int = 95) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

