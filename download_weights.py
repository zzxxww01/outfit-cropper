from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str | None
    note: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "florence": ModelSpec(
        name="florence",
        repo_id="microsoft/Florence-2-base",
        note="Open-vocabulary detection backbone for Step2.",
    ),
    "sam": ModelSpec(
        name="sam",
        repo_id="facebook/sam-vit-base",
        note="Segmentation model backbone for Step2.",
    ),
    "inpaint": ModelSpec(
        name="inpaint",
        repo_id="runwayml/stable-diffusion-inpainting",
        note="Inpainting model backbone for Step1.",
    ),
    "surya": ModelSpec(
        name="surya",
        repo_id=None,
        note="Surya weights are managed by surya-ocr runtime; run one warmup inference after install.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Phase1 model weights to checkpoints/"
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--models",
        type=str,
        default="surya,florence,sam,inpaint",
        help="Comma separated model names.",
    )
    parser.add_argument("--force", action="store_true", help="Force redownload.")
    return parser.parse_args()


def _download_hf_repo(repo_id: str, target_dir: Path, force: bool) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=not force,
    )


def main() -> int:
    args = parse_args()
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    selected = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for model_name in selected:
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            print(
                f"[skip] unknown model '{model_name}'. available={list(MODEL_SPECS.keys())}"
            )
            continue

        target = args.checkpoints_dir / spec.name
        target.mkdir(parents=True, exist_ok=True)
        print(f"[info] {spec.name}: {spec.note}")

        if spec.repo_id is None:
            print(f"[todo] no fixed repository configured for '{spec.name}'.")
            continue

        try:
            _download_hf_repo(spec.repo_id, target, force=args.force)
            print(f"[ok] downloaded '{spec.name}' -> {target}")
        except ImportError:
            print(
                "[error] huggingface-hub is not installed. Install requirements_gpu.txt first."
            )
            return 1
        except Exception as exc:
            print(f"[error] failed to download '{spec.name}': {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
