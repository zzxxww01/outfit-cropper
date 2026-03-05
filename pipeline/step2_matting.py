from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from schemas.models import GpuMetaItem
from utils.gpu_memory import clear_cuda_cache, log_gpu_memory
from utils.image_utils import (
    bbox_xyxy_to_1000,
    clamp_bbox_xyxy,
    composite_on_white,
    crop_rgb,
    expand_bbox_xyxy,
)
from utils.io_utils import ensure_dir

BBoxXYXY = Tuple[int, int, int, int]

# DeepFashion2 classes -> 8 ItemTypeEnum mapping
# 0: short_sleeved_shirt -> top
# 1: long_sleeved_shirt -> top
# 2: short_sleeved_outwear -> outerwear
# 3: long_sleeved_outwear -> outerwear
# 4: vest -> top
# 5: sling -> top
# 6: shorts -> bottom
# 7: trousers -> bottom
# 8: skirt -> bottom
# 9: short_sleeved_dress -> one_piece
# 10: long_sleeved_dress -> one_piece
# 11: vest_dress -> one_piece
# 12: sling_dress -> one_piece
DEEPFASHION2_VALID_CLASSES = set(range(13))


class MattingProcessor:
    """
    Phase 1 - Step 2/3 implementation
    Uses YOLOv8-Seg (DeepFashion2) for one-pass object detection and segmentation.
    Replaces the fragmented Florence-2 + SAM pipeline.
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
        self.device = device if device != "cpu" else "cpu"
        if self.device == "cuda":
            self.device = "0"  # Ultralytics typically takes "0" for cuda:0
        self.logger = logger or logging.getLogger(__name__)
        self.max_items = max_items
        self.min_box_area_ratio = min_box_area_ratio
        self.pad_ratio = pad_ratio
        self._yolo_model = None

    def load_models(self) -> None:
        if self._yolo_model is not None:
            return

        self.logger.info("Loading YOLOv8-Seg Model...")
        yolo_path = self.checkpoints_dir / "yolo" / "deepfashion2_yolov8s-seg.pt"
        if not yolo_path.exists():
            self.logger.warning(
                f"Weights not found at {yolo_path}. YOLO will throw an error if the model cannot be downloaded automatically."
            )

        self._yolo_model = YOLO(yolo_path)
        self.logger.info("Step2 YOLO models loaded.")
        log_gpu_memory(self.logger, prefix="After Step2 load")

    def unload_models(self) -> None:
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        clear_cuda_cache(self.logger, reason="after_step2")
        log_gpu_memory(self.logger, prefix="After Step2 unload")

    def extract_and_save_items(
        self, image: Image.Image, output_dir: Path
    ) -> List[GpuMetaItem]:
        self.load_models()
        ensure_dir(output_dir)

        if image is None:
            self.logger.warning("Step2 received None image input, skip.")
            return []
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Step2 expects PIL.Image.Image, got {type(image).__name__}."
            )

        image_rgb = image.convert("RGB")
        image_np = np.asarray(image_rgb, dtype=np.uint8)
        if image_np is None or image_np.ndim < 2:
            self.logger.warning("Step2 got invalid image array, skip.")
            return []
        height, width = image_np.shape[:2]

        min_area = width * height * self.min_box_area_ratio

        # Run inference
        results = self._yolo_model.predict(
            source=image_rgb,
            conf=0.4,
            device=self.device,
            verbose=False,
            retina_masks=True,  # Use high resolution masks
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or result.masks is None:
            self.logger.info("No candidates or masks found by YOLO.")
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()  # [N, H, W]

        # Filter valid boxes based on classes and min_area
        valid_indices = []
        for i in range(len(boxes_xyxy)):
            cls_id = int(classes[i])
            if cls_id not in DEEPFASHION2_VALID_CLASSES:
                continue

            x1, y1, x2, y2 = boxes_xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                valid_indices.append(i)

        # Sort by confidence descending and cap to max_items
        valid_indices = sorted(
            valid_indices, key=lambda i: confidences[i], reverse=True
        )[: self.max_items]

        items: List[GpuMetaItem] = []
        for index in valid_indices:
            box = boxes_xyxy[index]
            conf = confidences[index]
            mask_float = masks_data[index]

            # Resize the mask to original image size if it doesn't match
            if mask_float.shape != (height, width):
                mask_float = cv2.resize(
                    mask_float, (width, height), interpolation=cv2.INTER_LINEAR
                )

            binary_mask = (mask_float > 0.5).astype(np.uint8) * 255

            object_box = clamp_bbox_xyxy(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])), width, height
            )
            padded_box = expand_bbox_xyxy(
                object_box, width, height, pad_ratio=self.pad_ratio
            )

            composed = composite_on_white(image_np, binary_mask)
            crop = crop_rgb(composed, padded_box)
            if crop.size == 0:
                continue

            item_name = f"item_{len(items)}.jpg"
            item_path = output_dir / item_name
            Image.fromarray(crop).save(item_path, format="JPEG", quality=95)

            meta_item = GpuMetaItem(
                item_image_path=item_name,
                bbox=bbox_xyxy_to_1000(object_box, width, height),
                is_fallback=False,
                confidence=round(float(conf), 4),
            )
            items.append(meta_item)

        return items
