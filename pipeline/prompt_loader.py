from __future__ import annotations

from pathlib import Path


def resolve_prompt_path(prompts_dir: Path, prompt_version: str) -> Path:
    normalized = prompt_version.strip()
    if normalized.endswith(".txt"):
        return prompts_dir / normalized
    if not normalized.startswith("flatlay_"):
        normalized = f"flatlay_{normalized}"
    return prompts_dir / f"{normalized}.txt"


def load_prompt(prompts_dir: Path, prompt_version: str) -> str:
    prompt_path = resolve_prompt_path(prompts_dir, prompt_version)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt_text
