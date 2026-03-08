from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from pipeline.flatlay_mask_extractor import FlatlayMaskExtractor
from utils.io_utils import ensure_dir, read_image, write_json
from utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract outfit items from generated flatlay images using a pure mask-first pipeline."
    )
    parser.add_argument("--round-dir", type=Path, required=True, help="Directory containing outfit subfolders with flatlay.png")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <round-dir>_extract_mask")
    parser.add_argument("--only-outfit-id", type=str, default="")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument("--min-pad-px", type=int, default=24)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--minimal-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    round_dir = args.round_dir
    if not round_dir.exists():
        raise FileNotFoundError(f"Round directory does not exist: {round_dir}")

    output_dir = args.output_dir or Path(f"{round_dir}_extract_mask")
    ensure_dir(output_dir)

    logs_dir = Path("logs")
    ensure_dir(logs_dir)
    logger = setup_logger(
        name="segment_flatlay_mask",
        log_level=args.log_level,
        log_file=logs_dir / f"segment_flatlay_mask_{round_dir.name}.log",
    )

    extractor = FlatlayMaskExtractor(
        max_items=args.max_items,
        logger=logger,
        output_pad_ratio=args.pad_ratio,
        min_output_pad_px=args.min_pad_px,
    )

    outfit_dirs = sorted(
        [path for path in round_dir.iterdir() if path.is_dir() and (path / "flatlay.png").exists()],
        key=lambda path: path.name,
    )
    if args.only_outfit_id:
        outfit_dirs = [path for path in outfit_dirs if path.name == args.only_outfit_id]

    report: Dict[str, Any] = {
        "round_dir": str(round_dir),
        "output_dir": str(output_dir),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "pipeline": "mask_seg_flatlay",
        "pad_ratio": args.pad_ratio,
        "min_pad_px": args.min_pad_px,
        "items_per_outfit": {},
        "pair_group_counts": {"shoe_pair": 0, "accessory_pair": 0},
        "errors": [],
    }

    for outfit_dir in outfit_dirs:
        outfit_id = outfit_dir.name
        logger.info("Extracting outfit_id=%s", outfit_id)
        destination_dir = output_dir / outfit_id
        items_dir = destination_dir / "items"
        ensure_dir(items_dir)

        try:
            flatlay_path = outfit_dir / "flatlay.png"
            image = read_image(flatlay_path)
            result = extractor.extract_items(image)
            save_debug_artifacts = not args.minimal_output

            shutil.copy2(flatlay_path, destination_dir / "flatlay.png")
            source_path = outfit_dir / "source.jpg"
            if source_path.exists():
                shutil.copy2(source_path, destination_dir / "source.jpg")

            for item in result.items:
                item_png_path = items_dir / f"{item.item_id}.png"
                item.rgba_crop.save(item_png_path)
                if item.group_type in report["pair_group_counts"]:
                    report["pair_group_counts"][item.group_type] += 1
                if save_debug_artifacts:
                    item_white_path = items_dir / f"{item.item_id}_white.jpg"
                    item_mask_path = items_dir / f"{item.item_id}_mask.png"
                    item.white_crop.save(item_white_path, format="JPEG", quality=95)
                    Image.fromarray(item.mask).save(item_mask_path)

            if save_debug_artifacts:
                result.relayout_image.save(destination_dir / "relayout.png")
            meta = {
                "outfit_id": outfit_id,
                "source_image": "source.jpg" if source_path.exists() else "",
                "flatlay_image": "flatlay.png",
                "status": "ok",
                **result.to_dict(),
            }
            for item in meta["items"]:
                item_id = item["item_id"]
                item["image_path"] = f"items/{item_id}.png"
                if save_debug_artifacts:
                    item["white_bg_path"] = f"items/{item_id}_white.jpg"
                    item["mask_path"] = f"items/{item_id}_mask.png"
            write_json(destination_dir / "meta.json", meta)

            report["processed"] += 1
            report["succeeded"] += 1
            report["items_per_outfit"][outfit_id] = len(result.items)
            logger.info("Completed outfit_id=%s items=%s", outfit_id, len(result.items))
        except Exception as exc:  # noqa: BLE001
            report["processed"] += 1
            report["failed"] += 1
            report["errors"].append({"outfit_id": outfit_id, "message": str(exc)})
            logger.error("Failed outfit_id=%s: %s", outfit_id, exc)

    write_json(output_dir / "batch_report.json", report)
    logger.info(
        "Flatlay mask extraction round finished. succeeded=%s failed=%s",
        report["succeeded"],
        report["failed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
