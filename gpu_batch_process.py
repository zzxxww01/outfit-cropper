from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import List

from pipeline.step1_inpainting import InpaintingProcessor
from pipeline.step2_matting import MattingProcessor
from schemas.models import BatchErrorItem, BatchErrorReport, GpuMeta
from utils.io_utils import ensure_dir, list_image_files, read_image, safe_stem, write_json
from utils.logging_utils import setup_logger
from utils.retry import run_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 GPU batch processor for outfit-cropper.")
    parser.add_argument("--input-dir", type=Path, default=Path("input_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("gpu_output"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--only-outfit-id", type=str, default="", help="Process one file stem only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_dir = Path("logs")
    ensure_dir(logs_dir)
    logger = setup_logger(
        name="gpu_batch_process",
        log_level=args.log_level,
        log_file=logs_dir / "gpu_batch_process.log",
    )

    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return 1

    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoints_dir)

    image_paths = list_image_files(args.input_dir)
    if args.only_outfit_id:
        image_paths = [p for p in image_paths if safe_stem(p) == args.only_outfit_id]
    if not image_paths:
        logger.warning("No input images found under: %s", args.input_dir)
        return 0

    step1 = InpaintingProcessor(checkpoints_dir=args.checkpoints_dir, device=args.device, logger=logger)
    step2 = MattingProcessor(checkpoints_dir=args.checkpoints_dir, device=args.device, logger=logger)

    errors: List[BatchErrorItem] = []
    logger.info("Start Phase1 batch: %s image(s).", len(image_paths))

    for image_path in image_paths:
        outfit_id = safe_stem(image_path)
        outfit_dir = args.output_dir / outfit_id
        ensure_dir(outfit_dir)
        logger.info("Processing outfit_id=%s image=%s", outfit_id, image_path.name)
        step1_released = False

        try:
            original = read_image(image_path)
            repaired = run_with_retries(
                step1.run,
                original,
                max_retries=args.max_retries,
                logger=logger,
                step_name=f"{outfit_id}:step1_inpainting",
            )

            # Required red line from tech spec.
            step1.unload_models()
            step1_released = True

            items = run_with_retries(
                step2.extract_and_save_items,
                repaired,
                outfit_dir,
                max_retries=args.max_retries,
                logger=logger,
                step_name=f"{outfit_id}:step2_matting",
            )

            meta = GpuMeta(outfit_id=outfit_id, items=items)
            write_json(outfit_dir / "meta.json", meta.model_dump())
            logger.info("Completed outfit_id=%s items=%s", outfit_id, len(items))

        except Exception as exc:
            logger.error("Failed outfit_id=%s: %s", outfit_id, exc)
            logger.debug("Stack trace:\n%s", traceback.format_exc())
            fallback_meta = GpuMeta(outfit_id=outfit_id, items=[])
            write_json(outfit_dir / "meta.json", fallback_meta.model_dump())
            errors.append(
                BatchErrorItem(
                    outfit_id=outfit_id,
                    failed_step="phase1",
                    message=str(exc),
                    retries=args.max_retries,
                )
            )
        finally:
            # Keep strict memory hygiene when moving to next image.
            if not step1_released:
                step1.unload_models()

    step2.unload_models()

    report = BatchErrorReport(
        total_images=len(image_paths),
        failed_images=len(errors),
        errors=errors,
    )
    write_json(args.output_dir / "error_report.json", report.model_dump())
    logger.info("Phase1 batch finished. failed_images=%s", len(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
