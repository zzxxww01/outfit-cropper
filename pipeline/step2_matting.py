from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from schemas.models import GpuMetaItem
from utils.image_utils import (
    bbox_xyxy_to_1000,
    clamp_bbox_xyxy,
    composite_on_white,
    crop_rgb,
    expand_bbox_xyxy,
    mask_to_bbox_xyxy,
)
from utils.io_utils import ensure_dir

try:
    import cv2
except Exception:  # pragma: no cover - optional during local planning
    cv2 = None


BBoxXYXY = Tuple[int, int, int, int]


class MattingProcessor:
    """
    Phase 1 - Step 2/3 implementation shell.

    Current implementation uses robust CV fallbacks.
    TODO(Phase1 enhancement): replace detect/segment internals with Florence-2 + SAM.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        device: str = "cuda",
        logger: logging.Logger | None = None,
        max_items: int = 6,
        min_box_area_ratio: float = 0.015,
        pad_ratio: float = 0.1,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.max_items = max_items
        self.min_box_area_ratio = min_box_area_ratio
        self.pad_ratio = pad_ratio
        self._detector_model = None
        self._segment_model = None

    def load_models(self) -> None:
        if self._detector_model is not None and self._segment_model is not None:
            return
        # TODO(Phase1 enhancement): load Florence-2 and SAM models from checkpoints.
        self._detector_model = object()
        self._segment_model = object()
        self.logger.info("Step2 models loaded (placeholder mode).")

    def unload_models(self) -> None:
        if self._detector_model is not None:
            del self._detector_model
            self._detector_model = None
        if self._segment_model is not None:
            del self._segment_model
            self._segment_model = None

    def extract_and_save_items(self, image: Image.Image, output_dir: Path) -> List[GpuMetaItem]:
        self.load_models()
        ensure_dir(output_dir)

        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = image_rgb.shape[:2]

        candidate_boxes = self._detect_candidate_boxes(image_rgb)
        if not candidate_boxes:
            self.logger.info("No candidate boxes found.")
            return []

        items: List[GpuMetaItem] = []
        for index, box in enumerate(candidate_boxes[: self.max_items]):
            mask = self._segment_box(image_rgb, box)
            object_box = mask_to_bbox_xyxy(mask)
            if object_box is None:
                self.logger.debug("Skip candidate %s due to empty mask.", index)
                continue

            object_box = clamp_bbox_xyxy(object_box, width, height)
            padded_box = expand_bbox_xyxy(object_box, width, height, pad_ratio=self.pad_ratio)
            composed = composite_on_white(image_rgb, mask)
            crop = crop_rgb(composed, padded_box)
            if crop.size == 0:
                continue

            item_name = f"item_{len(items)}.jpg"
            item_path = output_dir / item_name
            Image.fromarray(crop).save(item_path, format="JPEG", quality=95)

            mask_area = float((mask > 0).sum())
            bbox_area = float((object_box[2] - object_box[0]) * (object_box[3] - object_box[1]))
            confidence = 0.0 if bbox_area <= 0 else min(1.0, mask_area / bbox_area)
            meta_item = GpuMetaItem(
                item_image_path=item_name,
                bbox=bbox_xyxy_to_1000(object_box, width, height),
                is_fallback=False,
                confidence=round(confidence, 4),
            )
            items.append(meta_item)

        return items

    def _detect_candidate_boxes(self, image_rgb: np.ndarray) -> List[BBoxXYXY]:
        h, w = image_rgb.shape[:2]
        if cv2 is None:
            return [self._fallback_center_box(w, h)]

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=60, threshold2=150)
        kernel = np.ones((5, 5), np.uint8)
        merged = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = h * w * self.min_box_area_ratio
        boxes: List[BBoxXYXY] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area:
                continue
            box = clamp_bbox_xyxy((x, y, x + bw, y + bh), w, h)
            boxes.append(box)

        boxes = self._nms(boxes, iou_threshold=0.5)
        if not boxes:
            boxes = [self._fallback_center_box(w, h)]
        return boxes

    def _segment_box(self, image_rgb: np.ndarray, box: BBoxXYXY) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = clamp_bbox_xyxy(box, w, h)
        if cv2 is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        try:
            cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            binary = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        except Exception:
            binary = np.zeros((h, w), dtype=np.uint8)
            binary[y1:y2, x1:x2] = 255
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return binary

    @staticmethod
    def _fallback_center_box(width: int, height: int) -> BBoxXYXY:
        x1 = int(width * 0.1)
        x2 = int(width * 0.9)
        y1 = int(height * 0.05)
        y2 = int(height * 0.95)
        return x1, y1, x2, y2

    @staticmethod
    def _nms(boxes: List[BBoxXYXY], iou_threshold: float = 0.5) -> List[BBoxXYXY]:
        if not boxes:
            return []

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        order = np.argsort(areas)[::-1]
        keep: List[BBoxXYXY] = []

        while len(order) > 0:
            i = int(order[0])
            current = boxes[i]
            keep.append(current)
            remaining = []
            for idx in order[1:]:
                if MattingProcessor._iou(current, boxes[int(idx)]) <= iou_threshold:
                    remaining.append(int(idx))
            order = np.array(remaining, dtype=np.int32)

        return keep

    @staticmethod
    def _iou(a: BBoxXYXY, b: BBoxXYXY) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

