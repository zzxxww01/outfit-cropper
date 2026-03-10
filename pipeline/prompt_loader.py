from __future__ import annotations

from pathlib import Path


ALLOWED_PROMPT_ALIASES = {
    "v1": "flatlay_v1.txt",
    "flatlay_v1": "flatlay_v1.txt",
    "flatlay_v1.txt": "flatlay_v1.txt",
}


def normalize_prompt_version(prompt_version: str) -> str:
    normalized = prompt_version.strip()
    try:
        return ALLOWED_PROMPT_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(ALLOWED_PROMPT_ALIASES))
        raise ValueError(
            f"Unsupported prompt version: {prompt_version!r}. Only prompt v1 is retained. Allowed values: {allowed}"
        ) from exc


def resolve_prompt_path(prompts_dir: Path, prompt_version: str) -> Path:
    return prompts_dir / normalize_prompt_version(prompt_version)


def load_prompt(prompts_dir: Path, prompt_version: str) -> str:
    prompt_path = resolve_prompt_path(prompts_dir, prompt_version)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt_text
