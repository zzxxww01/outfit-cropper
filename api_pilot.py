from __future__ import annotations

import argparse
import hashlib
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from pipeline.nano_banana_client import GeminiApiError, GeminiImageClient
from pipeline.prompt_loader import load_prompt
from pipeline.sample_selector import load_or_create_manifest
from utils.io_utils import ensure_dir, write_json
from utils.logging_utils import setup_logger
from utils.retry import run_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Nano Banana 2 prompt-iteration pilot on sampled outfit images."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("normal_1068807_1070000"))
    parser.add_argument("--output-dir", type=Path, default=Path("pilot_output"))
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=20260306)
    parser.add_argument("--round-id", type=str, default="round_001")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--only-outfit-id", type=str, default="")
    parser.add_argument("--prompt-version", type=str, default="v9")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-image-preview")
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--aspect-ratio", type=str, default="9:16")
    parser.add_argument("--image-size", type=str, default="2K")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-debug-artifacts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--minimal-output", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(outfit_dir: Path, resume: bool) -> bool:
    if not resume:
        return False
    return (outfit_dir / "flatlay.png").exists()


def write_review_stub(outfit_dir: Path, outfit_id: str, prompt_version: str) -> None:
    review_path = outfit_dir / "review.json"
    if review_path.exists():
        return
    write_json(
        review_path,
        {
            "outfit_id": outfit_id,
            "status": "pending_review",
            "prompt_version": prompt_version,
            "notes": "",
        },
    )


def main() -> int:
    load_dotenv()
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    ensure_dir(args.output_dir)
    round_dir = args.output_dir / args.round_id
    ensure_dir(round_dir)

    logs_dir = Path("logs")
    ensure_dir(logs_dir)
    logger = setup_logger(
        name="api_pilot",
        log_level=args.log_level,
        log_file=logs_dir / f"api_pilot_{args.round_id}.log",
    )

    manifest_path = args.manifest_path or (args.output_dir / "manifest.json")
    sample_paths = load_or_create_manifest(
        input_dir=args.input_dir,
        manifest_path=manifest_path,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if args.only_outfit_id:
        sample_paths = [path for path in sample_paths if path.stem == args.only_outfit_id]
        if not sample_paths:
            logger.error("Outfit id not found in manifest: %s", args.only_outfit_id)
            return 1

    prompt_text = load_prompt(args.prompts_dir, args.prompt_version)
    save_debug_artifacts = not args.minimal_output
    client = None
    if not args.dry_run:
        client = GeminiImageClient.from_environment(
            api_key_env=args.api_key_env,
            model=args.model,
            timeout_seconds=args.request_timeout,
            temperature=args.temperature,
        )

    report: Dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "round_id": args.round_id,
        "prompt_version": args.prompt_version,
        "model": args.model,
        "temperature": args.temperature,
        "sample_seed": args.sample_seed,
        "sample_size": len(sample_paths),
        "dry_run": args.dry_run,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }

    logger.info("Starting pilot round %s on %s image(s).", args.round_id, len(sample_paths))

    for image_path in sample_paths:
        outfit_id = image_path.stem
        outfit_dir = round_dir / outfit_id
        ensure_dir(outfit_dir)

        if should_skip(outfit_dir, resume=not args.no_resume):
            logger.info("Skipping outfit_id=%s because outputs already exist.", outfit_id)
            report["skipped"] += 1
            continue

        logger.info("Processing outfit_id=%s image=%s", outfit_id, image_path.name)
        shutil.copy2(image_path, outfit_dir / "source.jpg")
        if save_debug_artifacts:
            (outfit_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            write_review_stub(outfit_dir, outfit_id, args.prompt_version)
            write_json(
                outfit_dir / "request.json",
                {
                    "outfit_id": outfit_id,
                    "prompt_version": args.prompt_version,
                    "prompt_path": "prompt.txt",
                    "model": args.model,
                    "api_key_env": args.api_key_env,
                    "source_image_name": image_path.name,
                    "source_image_sha256": sha256_file(image_path),
                    "aspect_ratio": args.aspect_ratio,
                    "image_size": args.image_size,
                    "temperature": args.temperature,
                    "response_modalities": ["TEXT", "IMAGE"],
                },
            )

        if args.dry_run:
            report["processed"] += 1
            report["succeeded"] += 1
            logger.info("Prepared dry-run artifacts for outfit_id=%s", outfit_id)
            continue

        try:
            assert client is not None
            result = run_with_retries(
                client.generate_from_reference_image,
                prompt_text,
                image_path,
                aspect_ratio=args.aspect_ratio,
                image_size=args.image_size,
                temperature=args.temperature,
                max_retries=args.max_retries,
                logger=logger,
                step_name=f"{outfit_id}:nano_banana_generate",
            )
            if save_debug_artifacts:
                write_json(outfit_dir / "response.json", result.response_summary)
            result.save_first_image(outfit_dir / "flatlay.png")
            report["processed"] += 1
            report["succeeded"] += 1
            logger.info("Completed outfit_id=%s", outfit_id)
        except Exception as exc:  # noqa: BLE001
            report["processed"] += 1
            report["failed"] += 1
            error_record: Dict[str, Any] = {
                "outfit_id": outfit_id,
                "message": str(exc),
                "exception_type": type(exc).__name__,
            }
            if isinstance(exc, GeminiApiError):
                if exc.status_code is not None:
                    error_record["status_code"] = exc.status_code
                if save_debug_artifacts and exc.response_summary is not None:
                    write_json(outfit_dir / "response.json", exc.response_summary)
            if save_debug_artifacts:
                write_json(
                    outfit_dir / "error.json",
                    {
                        **error_record,
                        "stack_trace": traceback.format_exc(),
                    },
                )
            report["errors"].append(error_record)
            logger.error("Failed outfit_id=%s: %s", outfit_id, exc)
            logger.debug("Stack trace:\n%s", traceback.format_exc())

    write_json(round_dir / "batch_report.json", report)
    logger.info(
        "Pilot round finished. succeeded=%s failed=%s skipped=%s",
        report["succeeded"],
        report["failed"],
        report["skipped"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
