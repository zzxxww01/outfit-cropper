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

        # Load with explicit config to avoid cached module issues
        self._florence_processor = AutoProcessor.from_pretrained(
            florence_path, trust_remote_code=True
        )
        self._florence_model = AutoModelForCausalLM.from_pretrained(
            florence_path,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="eager",
            device_map=None,  # Don't use device_map, manually move to device
        ).to(self.device).eval()
        # Some Florence checkpoints ship generation_config with beam-only flags enabled.
        # Normalize to our single-beam setup to avoid noisy validation warnings.
        for cfg_owner in (
            self._florence_model,
            getattr(self._florence_model, "language_model", None),
        ):
            if cfg_owner is None:
                continue
            generation_config = getattr(cfg_owner, "generation_config", None)
            if generation_config is None:
                continue
            generation_config.early_stopping = False

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

        candidate_boxes = self._detect_candidate_boxes(image_rgb)
        if not candidate_boxes:
            self.logger.info("No candidate boxes found.")
            return []

        items: List[GpuMetaItem] = []
        for index, box in enumerate(candidate_boxes[: self.max_items]):
            try:
                mask = self._segment_box(image_rgb, box)
            except Exception as exc:
                self.logger.warning("Skip candidate %s due to SAM error: %s", index, exc)
                self.logger.debug("SAM candidate traceback", exc_info=True)
                continue

            if mask is None or mask.ndim != 2:
                self.logger.warning(
                    "Skip candidate %s due to invalid mask shape: %s",
                    index,
                    getattr(mask, "shape", None),
                )
                continue

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

        raw_inputs = self._florence_processor(
            text=prompt, images=image_rgb, return_tensors="pt"
        )

        model_inputs = {}
        for key, value in raw_inputs.items():
            if value is None or not isinstance(value, torch.Tensor):
                continue
            if key == "pixel_values":
                model_inputs[key] = value.to(self.device, dtype=torch.float16)
            else:
                model_inputs[key] = value.to(self.device)

        if model_inputs.get("pixel_values") is None or model_inputs.get("input_ids") is None:
            self.logger.warning(
                "Florence inputs missing required tensors. keys=%s",
                list(raw_inputs.keys()),
            )
            return []

        generate_base_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 1,
            "early_stopping": False,
            "use_cache": False,
            "pad_token_id": self._florence_processor.tokenizer.pad_token_id,
        }
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            generate_base_kwargs["attention_mask"] = attention_mask

        try:
            with torch.no_grad():
                generated_ids = self._florence_model.generate(**generate_base_kwargs)
        except Exception as exc:
            self.logger.warning("Florence generate failed (use_cache=False): %s", exc)
            self.logger.debug("Florence generate traceback", exc_info=True)
            return []

        if generated_ids is None:
            self.logger.warning("Florence generate returned None.")
            return []

        decoded = self._florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        if not decoded:
            self.logger.warning("Florence decode returned empty output.")
            return []

        generated_text = decoded[0]
        try:
            parsed_answer = self._florence_processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image_rgb.width, image_rgb.height),
            )
        except Exception as exc:
            self.logger.warning("Florence post-process failed: %s", exc)
            self.logger.debug("Florence post-process traceback", exc_info=True)
            return []
        if parsed_answer is None:
            self.logger.warning("Florence post-process returned None.")
            return []

        boxes = []
        min_area = image_rgb.width * image_rgb.height * self.min_box_area_ratio

        if isinstance(parsed_answer, dict) and "<OD>" in parsed_answer and "bboxes" in parsed_answer["<OD>"]:
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
        return boxes

    def _segment_box(self, image_rgb: Image.Image, box: BBoxXYXY) -> np.ndarray:
        empty_mask = np.zeros((image_rgb.height, image_rgb.width), dtype=np.uint8)

        # SAM expects input_boxes in format (batch_size, num_boxes, 4) with (x1, y1, x2, y2)
        # box is already (x1, y1, x2, y2), wrap it properly
        inputs = self._sam_processor(
            image_rgb,
            input_boxes=[[[box[0], box[1], box[2], box[3]]]],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        pred_masks = getattr(outputs, "pred_masks", None)
        iou_scores = getattr(outputs, "iou_scores", None)
        if pred_masks is None or iou_scores is None:
            self.logger.warning(
                "SAM returned empty outputs for box=%s (pred_masks=%s, iou_scores=%s).",
                box,
                pred_masks is not None,
                iou_scores is not None,
            )
            return empty_mask

        original_sizes = inputs.get("original_sizes")
        reshaped_input_sizes = inputs.get("reshaped_input_sizes")
        if original_sizes is None or reshaped_input_sizes is None:
            self.logger.warning(
                "SAM processor missing size tensors for box=%s (keys=%s).",
                box,
                list(inputs.keys()),
            )
            return empty_mask

        pred_masks_cpu = pred_masks.detach().cpu()
        iou_scores_cpu = iou_scores.detach().cpu()
        original_sizes_cpu = (
            original_sizes.detach().cpu()
            if isinstance(original_sizes, torch.Tensor)
            else original_sizes
        )
        reshaped_input_sizes_cpu = (
            reshaped_input_sizes.detach().cpu()
            if isinstance(reshaped_input_sizes, torch.Tensor)
            else reshaped_input_sizes
        )

        # Post-process masks to original image size
        masks = self._sam_processor.image_processor.post_process_masks(
            pred_masks_cpu,
            original_sizes_cpu,
            reshaped_input_sizes_cpu,
            return_tensors="pt",
        )

        if isinstance(masks, list):
            if not masks or masks[0] is None:
                self.logger.warning("SAM produced empty post-processed masks for box=%s.", box)
                return empty_mask
            masks = masks[0]
        if not isinstance(masks, torch.Tensor):
            self.logger.warning("Unsupported SAM mask type for box=%s: %s", box, type(masks))
            return empty_mask

        # Normalize to [num_masks, H, W].
        if masks.ndim == 4:
            masks = masks[0]
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0)
        elif masks.ndim != 3:
            self.logger.warning("Unexpected SAM mask ndim=%s for box=%s.", masks.ndim, box)
            return empty_mask

        # Pick the mask with highest IoU score
        scores = iou_scores_cpu
        while scores.ndim > 1:
            scores = scores[0]
        if scores.numel() == 0 or masks.shape[0] == 0:
            self.logger.warning("Empty SAM scores/masks for box=%s.", box)
            return empty_mask
        best_mask_idx = int(scores.argmax().item())
        if best_mask_idx >= masks.shape[0]:
            best_mask_idx = int(masks.shape[0] - 1)
        mask = masks[best_mask_idx].numpy()  # [H, W]

        if mask.shape != (image_rgb.height, image_rgb.width):
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            mask_img = mask_img.resize(
                (image_rgb.width, image_rgb.height),
                Image.Resampling.NEAREST,
            )
            return np.asarray(mask_img, dtype=np.uint8)

        # Convert to binary mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        return binary_mask

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
