from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from pipeline.item_classifier import FlatlayItemClassifier
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
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--classification-device", type=str, default="auto")
    parser.add_argument("--classification-batch-size", type=int, default=8)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--disable-stage2-rerank", action="store_true")
    parser.add_argument("--stage2-model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--stage2-device", type=str, default="auto")
    parser.add_argument("--stage2-batch-size", type=int, default=4)
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
    classifier = None
    if not args.skip_classification:
        classifier = FlatlayItemClassifier(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            batch_size=args.classification_batch_size,
            logger=logger,
            device=args.classification_device,
            stage2_enabled=not args.disable_stage2_rerank,
            stage2_model_name=args.stage2_model,
            stage2_device=args.stage2_device,
            stage2_batch_size=args.stage2_batch_size,
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
        "classification_enabled": not args.skip_classification,
        "classification_model": args.clip_model if not args.skip_classification else "",
        "classification_pretrained": args.clip_pretrained if not args.skip_classification else "",
        "classification_device": classifier.device if classifier is not None else "",
        "classification_stage2_enabled": bool(classifier is not None and classifier.stage2_reranker is not None),
        "classification_stage2_model": args.stage2_model if classifier is not None and classifier.stage2_reranker is not None else "",
        "items_per_outfit": {},
        "pair_group_counts": {"shoe_pair": 0, "accessory_pair": 0},
        "stage2_triggered_items": 0,
        "topology_reranked_items": 0,
        "decision_reason_counts": {},
        "class_counts": {
            "Outerwear": 0,
            "Top": 0,
            "Bottom": 0,
            "One_piece": 0,
            "Shoes": 0,
            "Bag": 0,
            "Accessories": 0,
            "Unknown": 0,
        },
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
            if classifier is not None and result.items:
                classifications = classifier.classify_items(items=result.items, flatlay_image=image)
                for item, classification in zip(result.items, classifications):
                    item.class_name = classification.class_name
                    item.class_confidence = classification.class_confidence
                    item.classification_stage = classification.classification_stage
                    item.top_scores = classification.top_scores
                    item.stage2_triggered = classification.stage2_triggered
                    item.topology_reranked = classification.topology_reranked
                    item.decision_reason = classification.decision_reason
                    item.shape_features = classification.shape_features
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
                report["class_counts"][item.class_name] += 1
                if item.stage2_triggered:
                    report["stage2_triggered_items"] += 1
                if item.topology_reranked:
                    report["topology_reranked_items"] += 1
                if item.decision_reason:
                    report["decision_reason_counts"][item.decision_reason] = (
                        report["decision_reason_counts"].get(item.decision_reason, 0) + 1
                    )
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
                "classification_model": args.clip_model if classifier is not None else "",
                "classification_pretrained": args.clip_pretrained if classifier is not None else "",
                "classification_device": classifier.device if classifier is not None else "",
                "classification_stage2_enabled": bool(classifier is not None and classifier.stage2_reranker is not None),
                "classification_stage2_model": args.stage2_model if classifier is not None and classifier.stage2_reranker is not None else "",
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
