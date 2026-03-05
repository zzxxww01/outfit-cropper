from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

from schemas.models import GpuMetaItem
from utils.gpu_memory import clear_cuda_cache, log_gpu_memory
from utils.image_utils import (
    bbox_xyxy_to_1000,
    clamp_bbox_xyxy,
    composite_on_white,
    crop_rgb,
    expand_bbox_xyxy,
    mask_to_bbox_xyxy,
)
from utils.io_utils import ensure_dir

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import SamModel, SamProcessor

BBoxXYXY = Tuple[int, int, int, int]


class MattingProcessor:
    """
    Phase 1 - Step 2/3 implementation
    Uses Florence-2 for object detection and SAM for segmentation.
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
        self._florence_model = None
        self._florence_processor = None
        self._sam_model = None
        self._sam_processor = None

    def load_models(self) -> None:
        if self._florence_model is not None and self._sam_model is not None:
            return

        self.logger.info("Loading Florence-2 Model...")
        florence_path = self.checkpoints_dir / "florence"
        if not florence_path.exists():
            florence_path = "microsoft/Florence-2-base"
        self._florence_processor = AutoProcessor.from_pretrained(
            florence_path, trust_remote_code=True
        )
        self._florence_model = (
            AutoModelForCausalLM.from_pretrained(
                florence_path,
                trust_remote_code=True,
                attn_implementation="eager"  # Disable SDPA to avoid compatibility issues
            )
            .to(self.device)
            .eval()
        )

        self.logger.info("Loading SAM Model...")
        sam_path = self.checkpoints_dir / "sam"
        if not sam_path.exists():
            sam_path = "facebook/sam-vit-base"
        self._sam_processor = SamProcessor.from_pretrained(sam_path)
        self._sam_model = SamModel.from_pretrained(sam_path).to(self.device).eval()

        self.logger.info("Step2 models loaded.")
        log_gpu_memory(self.logger, prefix="After Step2 load")

    def unload_models(self) -> None:
        if self._florence_model is not None:
            del self._florence_model
            self._florence_model = None
        if self._florence_processor is not None:
            del self._florence_processor
            self._florence_processor = None
        if self._sam_model is not None:
            del self._sam_model
            self._sam_model = None
        if self._sam_processor is not None:
            del self._sam_processor
            self._sam_processor = None
        clear_cuda_cache(self.logger, reason="after_step2")
        log_gpu_memory(self.logger, prefix="After Step2 unload")

    def extract_and_save_items(
        self, image: Image.Image, output_dir: Path
    ) -> List[GpuMetaItem]:
        self.load_models()
        ensure_dir(output_dir)

        image_rgb = image.convert("RGB")
        image_np = np.asarray(image_rgb, dtype=np.uint8)
        height, width = image_np.shape[:2]

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
            padded_box = expand_bbox_xyxy(
                object_box, width, height, pad_ratio=self.pad_ratio
            )
            composed = composite_on_white(image_np, mask)
            crop = crop_rgb(composed, padded_box)
            if crop.size == 0:
                continue

            item_name = f"item_{len(items)}.jpg"
            item_path = output_dir / item_name
            Image.fromarray(crop).save(item_path, format="JPEG", quality=95)

            mask_area = float((mask > 0).sum())
            bbox_area = float(
                (object_box[2] - object_box[0]) * (object_box[3] - object_box[1])
            )
            confidence = 0.0 if bbox_area <= 0 else min(1.0, mask_area / bbox_area)
            meta_item = GpuMetaItem(
                item_image_path=item_name,
                bbox=bbox_xyxy_to_1000(object_box, width, height),
                is_fallback=False,
                confidence=round(confidence, 4),
            )
            items.append(meta_item)

        return items

    def _detect_candidate_boxes(self, image_rgb: Image.Image) -> List[BBoxXYXY]:
        # Florence-2 task token must be used alone, no additional text
        prompt = "<OD>"

        inputs = self._florence_processor(
            text=prompt, images=image_rgb, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self._florence_processor.post_process_generation(
            generated_text, task=prompt, image_size=(image_rgb.width, image_rgb.height)
        )

        boxes = []
        min_area = image_rgb.width * image_rgb.height * self.min_box_area_ratio

        if "<OD>" in parsed_answer and "bboxes" in parsed_answer["<OD>"]:
            for bbox in parsed_answer["<OD>"]["bboxes"]:
                x1, y1, x2, y2 = bbox
                box = clamp_bbox_xyxy(
                    (int(x1), int(y1), int(x2), int(y2)),
                    image_rgb.width,
                    image_rgb.height,
                )
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area >= min_area:
                    boxes.append(box)

        boxes = self._nms(boxes, iou_threshold=0.5)
        if not boxes:
            boxes = [self._fallback_center_box(image_rgb.width, image_rgb.height)]
        return boxes

    def _segment_box(self, image_rgb: Image.Image, box: BBoxXYXY) -> np.ndarray:
        # SAM expects input_boxes as [[x1, y1, x2, y2]]
        inputs = self._sam_processor(
            image_rgb, input_boxes=[[[box]]], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        # Get the mask with highest IoU score
        masks = outputs.pred_masks.squeeze(1)
        scores = outputs.iou_scores.squeeze(1)
        best_mask_idx = scores.argmax()

        mask = masks[0, best_mask_idx].cpu().numpy()

        # Resize mask to original image size if needed
        if mask.shape != (image_rgb.height, image_rgb.width):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((image_rgb.width, image_rgb.height), Image.Resampling.NEAREST)
            mask = np.array(mask_img) / 255.0

        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        return binary_mask

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
