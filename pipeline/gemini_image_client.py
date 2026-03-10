from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

from PIL import Image


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


@dataclass
class GeneratedImagePart:
    mime_type: str
    data: bytes


@dataclass
class GeminiGenerateResult:
    response_json: Dict[str, Any]
    response_summary: Dict[str, Any]
    text_parts: List[str]
    image_parts: List[GeneratedImagePart]

    def save_first_image(self, output_path: Path) -> None:
        if not self.image_parts:
            raise ValueError("No generated image found in API response.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(BytesIO(self.image_parts[0].data))
        image.save(output_path, format="PNG")


class GeminiApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_summary: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_summary = response_summary


class GeminiImageClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.1-flash-image-preview",
        timeout_seconds: int = 120,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty.")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    @classmethod
    def from_environment(
        cls,
        *,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-3.1-flash-image-preview",
        timeout_seconds: int = 120,
        temperature: float = 0.2,
    ) -> "GeminiImageClient":
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env} is not set. Set it before running the pilot."
            )
        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    @property
    def endpoint(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def generate_from_reference_image(
        self,
        prompt: str,
        image_path: Path,
        *,
        aspect_ratio: str = "9:16",
        image_size: str = "2K",
        temperature: float | None = None,
    ) -> GeminiGenerateResult:
        payload = self._build_payload(
            prompt=prompt,
            image_path=image_path,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            temperature=self.temperature if temperature is None else temperature,
        )
        response_json = self._post_json(payload)
        return self._parse_response(response_json)

    def _build_payload(
        self,
        *,
        prompt: str,
        image_path: Path,
        aspect_ratio: str,
        image_size: str,
        temperature: float,
    ) -> Dict[str, Any]:
        image_bytes = image_path.read_bytes()
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": guess_mime_type(image_path),
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
            },
        }

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        curl_binary = shutil.which("curl.exe") or shutil.which("curl")
        if curl_binary:
            return self._post_json_with_curl(curl_binary, payload)

        return self._post_json_with_urllib(payload)

    def _post_json_with_curl(self, curl_binary: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gemini_api_") as temp_dir:
            temp_path = Path(temp_dir)
            payload_path = temp_path / "payload.json"
            response_path = temp_path / "response.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    curl_binary,
                    "-sS",
                    "-o",
                    str(response_path),
                    "-w",
                    "%{http_code}",
                    "-X",
                    "POST",
                    self.endpoint,
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    "Accept: application/json",
                    "--data-binary",
                    f"@{payload_path}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            status_text = result.stdout.strip()
            if not status_text.isdigit():
                raise GeminiApiError(
                    f"Gemini API curl request failed: {result.stderr.strip() or 'unknown curl error'}"
                )

            status_code = int(status_text)
            body = response_path.read_text(encoding="utf-8", errors="replace")

            try:
                response_json = json.loads(body)
            except json.JSONDecodeError as exc:
                raise GeminiApiError(
                    "Gemini API returned a non-JSON response.",
                    status_code=status_code or None,
                    response_summary={"raw_body": body[:4000]},
                ) from exc

            if status_code >= 400:
                raise GeminiApiError(
                    f"Gemini API request failed with status {status_code}.",
                    status_code=status_code,
                    response_summary=summarize_response(response_json),
                )

            return response_json

    def _post_json_with_urllib(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            response_summary: Dict[str, Any]
            try:
                response_json = json.loads(body)
                response_summary = summarize_response(response_json)
            except json.JSONDecodeError:
                response_summary = {"raw_body": body[:4000]}
            raise GeminiApiError(
                f"Gemini API request failed with status {exc.code}.",
                status_code=exc.code,
                response_summary=response_summary,
            ) from exc
        except error.URLError as exc:
            raise GeminiApiError(f"Gemini API request failed: {exc.reason}") from exc

    def _parse_response(self, response_json: Dict[str, Any]) -> GeminiGenerateResult:
        response_summary = summarize_response(response_json)
        image_parts = extract_image_parts(response_json)
        text_parts = extract_text_parts(response_json)
        if not image_parts:
            raise GeminiApiError(
                "Gemini API response did not contain an image part.",
                response_summary=response_summary,
            )
        return GeminiGenerateResult(
            response_json=response_json,
            response_summary=response_summary,
            text_parts=text_parts,
            image_parts=image_parts,
        )


def summarize_response(response_json: Dict[str, Any]) -> Dict[str, Any]:
    candidates = response_json.get("candidates", [])
    summary_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        summary_parts: List[Dict[str, Any]] = []
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                summary_parts.append({"type": "text", "preview": part["text"][:500]})
            elif "inlineData" in part:
                inline_data = part["inlineData"]
                summary_parts.append(
                    {
                        "type": "image",
                        "mime_type": inline_data.get("mimeType", "application/octet-stream"),
                        "base64_length": len(inline_data.get("data", "")),
                    }
                )
        summary_candidates.append(
            {
                "finish_reason": candidate.get("finishReason"),
                "parts": summary_parts,
            }
        )
    return {
        "prompt_feedback": response_json.get("promptFeedback"),
        "usage_metadata": response_json.get("usageMetadata"),
        "candidates": summary_candidates,
    }


def extract_text_parts(response_json: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for candidate in response_json.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return texts


def extract_image_parts(response_json: Dict[str, Any]) -> List[GeneratedImagePart]:
    images: List[GeneratedImagePart] = []
    for candidate in response_json.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline_data = part.get("inlineData")
            if not inline_data:
                continue
            data = inline_data.get("data")
            if not data:
                continue
            images.append(
                GeneratedImagePart(
                    mime_type=inline_data.get("mimeType", "application/octet-stream"),
                    data=base64.b64decode(data),
                )
            )
    return images
