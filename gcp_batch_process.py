from __future__ import annotations

import argparse
from pathlib import Path

from utils.io_utils import ensure_dir, read_json, write_json
from utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 local batch processor (TODO placeholder).")
    parser.add_argument("--gpu-output-dir", type=Path, default=Path("gpu_output"))
    parser.add_argument("--result-path", type=Path, default=Path("results/result.todo.json"))
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("gcp_batch_process", log_level=args.log_level, log_file=Path("logs/gcp_batch_process.log"))

    if not args.gpu_output_dir.exists():
        logger.error("gpu_output directory not found: %s", args.gpu_output_dir)
        return 1

    logger.warning("TODO(Phase2): Gemini classification is not implemented in this milestone.")
    outfits = []
    for outfit_dir in sorted(args.gpu_output_dir.iterdir()):
        if not outfit_dir.is_dir():
            continue
        meta_path = outfit_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        outfits.append(
            {
                "outfit_id": meta.get("outfit_id", outfit_dir.name),
                "items": [],
                "status": "todo_phase2",
                "message": "TODO(Phase2): classify item_type with Gemini and assemble final result.",
            }
        )

    ensure_dir(args.result_path.parent)
    write_json(
        args.result_path,
        {
            "status": "todo_phase2",
            "model": args.model,
            "outfits": outfits,
        },
    )
    logger.info("Generated placeholder result: %s", args.result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

