from __future__ import annotations

from pathlib import Path
from typing import Any


class GeminiClassifier:
    """
    TODO(Phase2): implement Vertex AI Gemini structured classification.
    """

    def __init__(self, project: str, location: str, model_name: str = "gemini-2.0-flash") -> None:
        self.project = project
        self.location = location
        self.model_name = model_name

    def classify_outfit(self, outfit_dir: Path, original_image: Path) -> Any:
        raise NotImplementedError(
            "TODO(Phase2): Gemini structured output classification is not implemented yet."
        )

