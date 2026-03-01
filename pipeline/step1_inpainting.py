from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from utils.gpu_memory import clear_cuda_cache, log_gpu_memory

try:
    import cv2
except Exception:  # pragma: no cover - optional during local planning
    cv2 = None


class InpaintingProcessor:
    """
    Phase 1 - Step 1 placeholder implementation.

    Real model wiring (Surya OCR + SD/PowerPaint) is intentionally isolated.
    This class already enforces the memory-release contract before Step 2.
    """

    def __init__(self, checkpoints_dir: Path, device: str = "cuda", logger: logging.Logger | None = None) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self._ocr_model = None
        self._inpaint_model = None

    def load_models(self) -> None:
        if self._ocr_model is not None and self._inpaint_model is not None:
            return

        # TODO(Phase1 enhancement): replace placeholders with real Surya and Inpainting model objects.
        self._ocr_model = object()
        self._inpaint_model = object()
        self.logger.info("Step1 models loaded (placeholder mode).")
        log_gpu_memory(self.logger, prefix="After Step1 load")

    def run(self, image: Image.Image) -> Image.Image:
        self.load_models()
        image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)

        if cv2 is None:
            self.logger.warning("OpenCV is unavailable; Step1 falls back to no-op.")
            return Image.fromarray(image_np)

        text_mask = self._build_text_like_mask(image_np)
        repaired = self._inpaint(image_np, text_mask)
        return Image.fromarray(repaired)

    def unload_models(self) -> None:
        # Hard requirement from tech spec: explicit del + empty cache between Step1 and Step2.
        if self._ocr_model is not None:
            del self._ocr_model
            self._ocr_model = None
        if self._inpaint_model is not None:
            del self._inpaint_model
            self._inpaint_model = None
        clear_cuda_cache(self.logger, reason="step1_to_step2")
        log_gpu_memory(self.logger, prefix="After Step1 unload")

    @staticmethod
    def _build_text_like_mask(image_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        high_sat = hsv[:, :, 1] > 130
        bright = hsv[:, :, 2] > 120
        mask = np.where(high_sat & bright, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    @staticmethod
    def _inpaint(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return image_rgb
        return cv2.inpaint(image_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

