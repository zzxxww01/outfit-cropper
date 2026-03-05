from __future__ import annotations

import logging
from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from utils.gpu_memory import clear_cuda_cache, log_gpu_memory

import torch
from diffusers import AutoPipelineForInpainting
from surya.detection import DetectionPredictor

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# Surya currently emits this deprecation warning from internal calls.
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)


class InpaintingProcessor:
    """
    Phase 1 - Step 1: Detects text using Surya OCR and inpaints using Stable Diffusion.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        device: str = "cuda",
        logger: logging.Logger | None = None,
    ) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self._ocr_predictor = None
        self._inpaint_model = None

    def load_models(self) -> None:
        if self._ocr_predictor is not None and self._inpaint_model is not None:
            return

        self.logger.info("Loading Surya OCR Detection Model...")
        # DetectionPredictor handles model loading automatically
        self._ocr_predictor = DetectionPredictor(device=self.device)

        self.logger.info("Loading Stable Diffusion Inpainting Model...")
        inpaint_path = self.checkpoints_dir / "inpaint"
        if not inpaint_path.exists():
            # Fallback to loading from HF directly if not downloaded to local path
            inpaint_path = "runwayml/stable-diffusion-inpainting"

        self._inpaint_model = AutoPipelineForInpainting.from_pretrained(
            inpaint_path, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

        self.logger.info("Step1 models loaded.")
        log_gpu_memory(self.logger, prefix="After Step1 load")

    def run(self, image: Image.Image) -> Image.Image:
        self.load_models()
        image_rgb = image.convert("RGB")
        image_np = np.asarray(image_rgb, dtype=np.uint8)

        text_mask = self._build_text_like_mask(image_rgb)

        # If no text found, return original
        if np.max(text_mask) == 0:
            return image_rgb

        mask_image = Image.fromarray(text_mask)
        repaired_image = self._inpaint(image_rgb, mask_image)
        return repaired_image

    def unload_models(self) -> None:
        # Hard requirement from tech spec: explicit del + empty cache between Step1 and Step2.
        if self._ocr_predictor is not None:
            del self._ocr_predictor
            self._ocr_predictor = None
        if self._inpaint_model is not None:
            del self._inpaint_model
            self._inpaint_model = None

        clear_cuda_cache(self.logger, reason="step1_to_step2")
        log_gpu_memory(self.logger, prefix="After Step1 unload")

    def _build_text_like_mask(self, image_rgb: Image.Image) -> np.ndarray:
        # Run Surya detection - DetectionPredictor is callable
        predictions = self._ocr_predictor([image_rgb])
        pred = predictions[0]

        width, height = image_rgb.size
        mask = np.zeros((height, width), dtype=np.uint8)

        for bbox_det in pred.bboxes:
            # Use polygon if available, otherwise use bbox
            polygon = bbox_det.polygon
            if cv2 is not None and polygon is not None:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            elif cv2 is not None:
                # Fallback to bbox if polygon not available
                x1, y1, x2, y2 = bbox_det.bbox
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

        # Dilate mask slightly to cover text edges well
        if cv2 is not None and np.max(mask) > 0:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def _inpaint(self, image_rgb: Image.Image, mask_image: Image.Image) -> Image.Image:
        # Resize image and mask to suitable sizes for SD if needed, or rely on auto-resizing
        # RunwayML SD Inpainting is typically 512x512, diffusers usually handles resizing natively
        generator = torch.Generator(device=self.device).manual_seed(42)

        output = self._inpaint_model(
            prompt="high quality, original background, seamless",
            negative_prompt="text, watermark, logo, bad anatomy",
            image=image_rgb,
            mask_image=mask_image,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        # Ensure output is same size as input if pipeline resized it
        if output.size != image_rgb.size:
            output = output.resize(image_rgb.size, Image.Resampling.LANCZOS)

        return output
