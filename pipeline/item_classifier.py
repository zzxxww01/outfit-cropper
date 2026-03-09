from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from utils.image_utils import clamp_bbox_xyxy, expand_bbox_xyxy

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:  # pragma: no cover - handled at runtime
    AutoModel = None
    AutoProcessor = None


CLASS_NAMES: tuple[str, ...] = (
    "Outerwear",
    "Top",
    "Bottom",
    "One_piece",
    "Shoes",
    "Bag",
    "Accessories",
)

AMBIGUOUS_CLASSES = {"Shoes", "Bag", "Accessories"}
CLOTHING_CLASS_NAMES: tuple[str, ...] = ("Outerwear", "Top", "Bottom", "One_piece")

CLASS_PROMPTS: Dict[str, tuple[str, ...]] = {
    "Outerwear": (
        "a product photo of an outerwear garment such as a coat, jacket, blazer, trench coat, or cardigan",
        "an outerwear layer worn over a shirt, blouse, knit top, or dress",
        "an isolated flat lay of a coat, jacket, blazer, trench coat, or cardigan on white background",
    ),
    "Top": (
        "a product photo of a top garment such as a t-shirt, shirt, blouse, sweater, hoodie, tank top, camisole, or vest",
        "an isolated flat lay of a shirt, blouse, sweater, hoodie, tank top, or other upper-body top on white background",
        "a fashion item worn only on the upper body with sleeves, shoulder straps, arm openings, neckline, or collar, and not connected to a skirt or pants",
    ),
    "Bottom": (
        "a product photo of a bottom garment such as pants, jeans, trousers, shorts, or skirt",
        "an isolated flat lay of pants, jeans, shorts, skirt, or an asymmetrical layered skirt on white background",
        "a fashion item worn on the lower body with a waistband or waist opening and no sleeves",
    ),
    "One_piece": (
        "a product photo of a one-piece garment such as a dress, jumpsuit, or romper",
        "an isolated flat lay of a dress or jumpsuit with the top and skirt or pants connected as one garment",
        "a fashion item that is a single connected garment for both upper and lower body",
    ),
    "Shoes": (
        "a product photo of a pair of shoes such as sneakers, boots, sandals, or leather shoes",
        "an isolated flat lay of footwear on white background",
        "a fashion item that is a pair of shoes",
    ),
    "Bag": (
        "a product photo of a bag such as a shoulder bag, crossbody bag, handbag, or backpack",
        "an isolated flat lay of a fashion bag on white background",
        "a fashion accessory that is a bag",
    ),
    "Accessories": (
        "a product photo of a fashion accessory such as a hat, belt, scarf, sunglasses, necklace, or earrings",
        "an isolated flat lay of a fashion accessory on white background",
        "a core styling accessory like hat, belt, scarf, sunglasses, necklace, or earrings",
    ),
}

CLOTHING_REFINE_PROMPTS: Dict[str, tuple[str, ...]] = {
    "Outerwear": (
        "an outerwear layer such as a jacket, coat, blazer, trench coat, or cardigan that is worn over another garment",
        "a clothing item that functions as an outer layer",
    ),
    "Top": (
        "an upper-body garment such as a shirt, blouse, sweater, hoodie, tank top, camisole, or vest with shoulders, sleeves, armholes, collar, or neckline",
        "a separate top garment for the upper body only",
    ),
    "Bottom": (
        "a lower-body garment such as pants, jeans, shorts, or skirt with a waistband at the top and no sleeves or collar",
        "a separate bottom garment for the lower body, including asymmetrical and layered skirts",
    ),
    "One_piece": (
        "a one-piece garment such as a dress, jumpsuit, or romper with the upper and lower parts connected together",
        "a single garment that combines top and bottom in one piece",
    ),
}

SIGLIP_CLOTHING_PROMPTS: Dict[str, tuple[str, ...]] = {
    "Outerwear": (
        "a standalone outerwear piece such as a jacket, coat, blazer, trench coat, or cardigan that layers over another garment",
        "an upper-body outer layer with sleeves, lapels, or a front opening",
    ),
    "Top": (
        "a standalone upper-body top such as a shirt, blouse, sweater, hoodie, tank top, camisole, or vest",
        "a garment for the upper body only, with neckline, straps, shoulders, sleeves, or armholes, but no skirt section",
    ),
    "Bottom": (
        "a standalone lower-body garment such as a skirt, pants, jeans, trousers, or shorts, with waistband at the top and no bodice or shoulder section",
        "a skirt or other bottom garment worn only on the lower body",
    ),
    "One_piece": (
        "a dress, jumpsuit, or romper with an upper-body bodice and a lower-body skirt or pants connected in one garment",
        "a one-piece garment that includes neckline or shoulder area and a connected lower-body section",
    ),
}


@dataclass
class ItemClassification:
    class_name: str
    class_confidence: float
    classification_stage: str
    top_scores: Dict[str, float]
    stage2_triggered: bool = False
    topology_reranked: bool = False
    decision_reason: str = ""
    shape_features: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "class_name": self.class_name,
            "class_confidence": round(self.class_confidence, 6),
            "classification_stage": self.classification_stage,
            "top_scores": {name: round(score, 6) for name, score in self.top_scores.items()},
        }
        if self.stage2_triggered:
            data["stage2_triggered"] = True
        if self.topology_reranked:
            data["topology_reranked"] = True
        if self.decision_reason:
            data["decision_reason"] = self.decision_reason
        if self.shape_features:
            data["shape_features"] = {
                name: round(value, 6) for name, value in self.shape_features.items()
            }
        return data


class SiglipClothingReranker:
    def __init__(
        self,
        *,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "auto",
        batch_size: int = 4,
        cache_dir: Path | None = None,
        logger: Any | None = None,
    ) -> None:
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError("transformers is required for SigLIP reranking.")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = max(1, batch_size)
        self.cache_dir = cache_dir or Path(".cache") / "huggingface"
        self.logger = logger
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model = AutoModel.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model.to(self.device)
        self.model.eval()
        self.prompts = self._build_prompt_list(SIGLIP_CLOTHING_PROMPTS)
        self.prompt_labels = [
            class_name
            for class_name in CLOTHING_CLASS_NAMES
            for _prompt in SIGLIP_CLOTHING_PROMPTS[class_name]
        ]

        if self.logger is not None:
            self.logger.info(
                "Loaded SigLIP reranker model=%s device=%s",
                self.model_name,
                self.device,
            )

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_prompt_list(self, prompt_map: Dict[str, tuple[str, ...]]) -> List[str]:
        prompts: List[str] = []
        for class_name in CLOTHING_CLASS_NAMES:
            prompts.extend(prompt_map[class_name])
        return prompts

    def score_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, len(CLOTHING_CLASS_NAMES)), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch_images = images[start:start + self.batch_size]
                inputs = self.processor(
                    text=self.prompts,
                    images=list(batch_images),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                prompt_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
                all_probs.append(self._aggregate_prompt_probs(prompt_probs))
        return np.concatenate(all_probs, axis=0)

    def _aggregate_prompt_probs(self, prompt_probs: np.ndarray) -> np.ndarray:
        class_probs = np.zeros((prompt_probs.shape[0], len(CLOTHING_CLASS_NAMES)), dtype=np.float32)
        for class_index, class_name in enumerate(CLOTHING_CLASS_NAMES):
            prompt_indexes = [
                idx for idx, label in enumerate(self.prompt_labels)
                if label == class_name
            ]
            class_probs[:, class_index] = np.mean(prompt_probs[:, prompt_indexes], axis=1)
        denom = class_probs.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-8)
        return class_probs / denom


class FlatlayItemClassifier:
    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 8,
        cache_dir: Path | None = None,
        logger: Any | None = None,
        device: str = "auto",
        stage2_model_name: str = "google/siglip-base-patch16-224",
        stage2_enabled: bool = True,
        stage2_batch_size: int = 4,
        stage2_device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = max(1, batch_size)
        self.cache_dir = cache_dir or Path(".cache") / "openclip"
        self.logger = logger
        self.device = self._resolve_device(device)
        self.stage2_enabled = stage2_enabled

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            cache_dir=str(self.cache_dir),
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_features = self._build_text_features()
        self.clothing_text_features = self._build_text_features(
            prompt_map=CLOTHING_REFINE_PROMPTS,
            class_names=CLOTHING_CLASS_NAMES,
        )
        self.stage2_reranker = self._build_stage2_reranker(
            stage2_enabled=stage2_enabled,
            stage2_model_name=stage2_model_name,
            stage2_batch_size=stage2_batch_size,
            stage2_device=stage2_device,
        )

        if self.logger is not None:
            self.logger.info(
                "Loaded OpenCLIP classifier model=%s pretrained=%s device=%s stage2=%s",
                self.model_name,
                self.pretrained,
                self.device,
                bool(self.stage2_reranker),
            )

    def _build_stage2_reranker(
        self,
        *,
        stage2_enabled: bool,
        stage2_model_name: str,
        stage2_batch_size: int,
        stage2_device: str,
    ) -> SiglipClothingReranker | None:
        if not stage2_enabled:
            return None
        try:
            return SiglipClothingReranker(
                model_name=stage2_model_name,
                batch_size=stage2_batch_size,
                device=stage2_device,
                logger=self.logger,
            )
        except Exception as exc:  # noqa: BLE001
            if self.logger is not None:
                self.logger.warning("SigLIP reranker unavailable: %s", exc)
            return None

    def classify_items(
        self,
        *,
        items: Sequence[Any],
        flatlay_image: Image.Image,
    ) -> List[ItemClassification]:
        if not items:
            return []

        crop_images = [self._prepare_crop_image(item.rgba_crop) for item in items]
        crop_probs = self._score_images(crop_images)

        results: List[ItemClassification] = []
        for index, item in enumerate(items):
            probs = crop_probs[index]
            top_class, top_conf, margin = self._prediction_stats(probs)
            stage = "crop_only"
            decision_reason = "clip_primary"
            stage2_triggered = False
            topology_reranked = False

            if self._should_use_context(top_class, top_conf, margin):
                context_image = self._build_context_crop(flatlay_image, item.bbox_xyxy)
                context_probs = self._score_images([context_image])[0]
                probs = (probs * 0.55) + (context_probs * 0.45)
                stage = "crop_plus_context"
                top_class, top_conf, margin = self._prediction_stats(probs)
                decision_reason = "clip_context"

            probs = self._refine_clothing_prediction(
                probs=probs,
                item=items[index],
                crop_image=crop_images[index],
                top_class=top_class,
            )
            pre_stage2_probs = probs.copy()

            if self._should_trigger_stage2(probs):
                reranked = self._apply_stage2_rerank(
                    probs=probs,
                    item=items[index],
                    crop_image=crop_images[index],
                    flatlay_image=flatlay_image,
                )
                if reranked is not None:
                    probs = reranked
                    stage2_triggered = True
                    stage = "crop_context_siglip"
                    decision_reason = "siglip_clothing_rerank"

            shape_features = self._shape_features(item.rgba_crop)
            if shape_features is not None:
                updated = self._apply_topology_rerank(
                    probs,
                    shape_features=shape_features,
                    pre_stage2_probs=pre_stage2_probs,
                )
                if updated is not None:
                    probs, topology_reason = updated
                    topology_reranked = True
                    decision_reason = topology_reason

            class_name, class_conf, finalize_reason = self._finalize_prediction(probs)
            if finalize_reason:
                decision_reason = finalize_reason

            results.append(
                ItemClassification(
                    class_name=class_name,
                    class_confidence=class_conf,
                    classification_stage=stage,
                    top_scores=self._top_scores(probs),
                    stage2_triggered=stage2_triggered,
                    topology_reranked=topology_reranked,
                    decision_reason=decision_reason,
                    shape_features=shape_features,
                )
            )
        return results

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_text_features(
        self,
        *,
        prompt_map: Dict[str, tuple[str, ...]] | None = None,
        class_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        prompt_map = prompt_map or CLASS_PROMPTS
        class_names = class_names or CLASS_NAMES
        class_vectors: List[torch.Tensor] = []
        with torch.no_grad():
            for class_name in class_names:
                prompts = prompt_map[class_name]
                tokens = self.tokenizer(list(prompts)).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_feature = text_features.mean(dim=0)
                mean_feature = mean_feature / mean_feature.norm()
                class_vectors.append(mean_feature)
        return torch.stack(class_vectors, dim=0)

    def _score_images(
        self,
        images: Sequence[Image.Image],
        *,
        text_features: torch.Tensor | None = None,
    ) -> np.ndarray:
        if not images:
            feature_count = len(CLASS_NAMES) if text_features is None else int(text_features.shape[0])
            return np.empty((0, feature_count), dtype=np.float32)
        if text_features is None:
            text_features = self.text_features

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch_images = images[start:start + self.batch_size]
                tensor_batch = torch.stack([self.preprocess(image) for image in batch_images], dim=0).to(self.device)
                image_features = self.model.encode_image(tensor_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.T
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                all_probs.append(probs.astype(np.float32))
        return np.concatenate(all_probs, axis=0)

    def _prepare_crop_image(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white, rgba)
        return composited.convert("RGB")

    def _build_silhouette_image(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba, dtype=np.uint8)[..., 3]
        silhouette = np.full((alpha.shape[0], alpha.shape[1], 3), 255, dtype=np.uint8)
        silhouette[alpha > 0] = 0
        return Image.fromarray(silhouette, mode="RGB")

    def _build_context_crop(
        self,
        flatlay_image: Image.Image,
        bbox_xyxy: Sequence[int],
    ) -> Image.Image:
        width, height = flatlay_image.size
        bbox = clamp_bbox_xyxy(tuple(int(v) for v in bbox_xyxy), width, height)
        context_bbox = expand_bbox_xyxy(bbox, width, height, pad_ratio=0.18)
        return flatlay_image.crop(context_bbox).convert("RGB")

    def _should_use_context(self, class_name: str, confidence: float, margin: float) -> bool:
        if confidence < 0.42:
            return True
        if class_name in AMBIGUOUS_CLASSES:
            return True
        if class_name in {"Outerwear", "One_piece"} and margin < 0.05:
            return True
        return False

    def _refine_clothing_prediction(
        self,
        *,
        probs: np.ndarray,
        item: Any,
        crop_image: Image.Image,
        top_class: str,
    ) -> np.ndarray:
        if top_class not in CLOTHING_CLASS_NAMES:
            return probs

        clothing_probs = self._score_images([crop_image], text_features=self.clothing_text_features)[0]
        clothing_conf = float(np.max(clothing_probs))
        if clothing_conf < 0.55:
            return probs

        refined = probs.copy()
        clothing_mass = float(sum(refined[CLASS_NAMES.index(name)] for name in CLOTHING_CLASS_NAMES))
        clothing_mass = max(clothing_mass, 0.65)
        for name in CLOTHING_CLASS_NAMES:
            refined[CLASS_NAMES.index(name)] = 0.0
        for index, name in enumerate(CLOTHING_CLASS_NAMES):
            refined[CLASS_NAMES.index(name)] = clothing_mass * float(clothing_probs[index])
        refined = refined / refined.sum()
        return refined

    def _should_trigger_stage2(self, probs: np.ndarray) -> bool:
        if self.stage2_reranker is None:
            return False

        top_class, _top_conf, _margin = self._prediction_stats(probs)
        bottom_score = float(probs[CLASS_NAMES.index("Bottom")])
        top_score = float(probs[CLASS_NAMES.index("Top")])
        one_piece_score = float(probs[CLASS_NAMES.index("One_piece")])
        outerwear_score = float(probs[CLASS_NAMES.index("Outerwear")])

        if top_class in CLOTHING_CLASS_NAMES:
            if top_class == "Outerwear":
                return False
            if top_class == "One_piece" and bottom_score >= 0.18:
                return True
            if top_class == "Top" and bottom_score >= 0.12:
                return True
            if abs(bottom_score - one_piece_score) < 0.12:
                return True
            if abs(bottom_score - top_score) < 0.10 and max(bottom_score, top_score) >= 0.14:
                return True
            if abs(outerwear_score - top_score) < 0.08 and max(outerwear_score, top_score) >= 0.20:
                return True
        if top_class == "Accessories":
            return False
        if top_class == "Bag" or top_class == "Shoes":
            return False
        if top_class == "Outerwear" and bottom_score >= 0.18:
            return True
        return False

    def _apply_stage2_rerank(
        self,
        *,
        probs: np.ndarray,
        item: Any,
        crop_image: Image.Image,
        flatlay_image: Image.Image,
    ) -> np.ndarray | None:
        if self.stage2_reranker is None:
            return None

        silhouette_image = self._build_silhouette_image(item.rgba_crop)
        context_image = self._build_context_crop(flatlay_image, item.bbox_xyxy)
        stage2_probs = self.stage2_reranker.score_images([crop_image, silhouette_image, context_image])
        crop_clothing = stage2_probs[0]
        silhouette_clothing = stage2_probs[1]
        context_clothing = stage2_probs[2]
        combined_clothing = (crop_clothing * 0.58) + (silhouette_clothing * 0.24) + (context_clothing * 0.18)
        combined_clothing = combined_clothing / max(float(combined_clothing.sum()), 1e-8)

        refined = probs.copy()
        clothing_mass = float(sum(refined[CLASS_NAMES.index(name)] for name in CLOTHING_CLASS_NAMES))
        clothing_mass = max(clothing_mass, 0.68)
        for name in CLOTHING_CLASS_NAMES:
            refined[CLASS_NAMES.index(name)] = 0.0
        for index, name in enumerate(CLOTHING_CLASS_NAMES):
            refined[CLASS_NAMES.index(name)] = clothing_mass * float(combined_clothing[index])
        refined = refined / refined.sum()
        return refined

    def _apply_topology_rerank(
        self,
        probs: np.ndarray,
        *,
        shape_features: Dict[str, float],
        pre_stage2_probs: np.ndarray,
    ) -> tuple[np.ndarray, str] | None:
        bottom_index = CLASS_NAMES.index("Bottom")
        top_index = CLASS_NAMES.index("Top")
        one_piece_index = CLASS_NAMES.index("One_piece")

        bottom_score = float(probs[bottom_index])
        top_score = float(probs[top_index])
        one_piece_score = float(probs[one_piece_index])
        outerwear_index = CLASS_NAMES.index("Outerwear")
        pre_bottom_score = float(pre_stage2_probs[bottom_index])
        pre_top_score = float(pre_stage2_probs[top_index])
        pre_one_piece_score = float(pre_stage2_probs[one_piece_index])
        pre_outerwear_score = float(pre_stage2_probs[outerwear_index])

        upper_body_evidence = (
            shape_features["top_center_gap"] >= 0.19
            or shape_features["top_segments"] >= 1.32
        )
        strong_upper_body_evidence = (
            shape_features["top_center_gap"] >= 0.28
            and shape_features["top_segments"] >= 1.80
        )
        skirt_evidence = (
            shape_features["top_center_gap"] <= 0.16
            and shape_features["top_segments"] <= 1.24
            and shape_features["mid_ratio"] >= shape_features["upper_ratio"] - 0.03
            and shape_features["lower_ratio"] >= shape_features["mid_ratio"] - 0.03
            and shape_features["aspect"] >= 1.12
        )

        top_class, _top_conf, _margin = self._prediction_stats(probs)
        refined = probs.copy()
        if pre_outerwear_score >= 0.22 and pre_outerwear_score + 0.05 >= max(pre_top_score, pre_bottom_score, pre_one_piece_score):
            return None
        if (
            top_class == "Bottom"
            and upper_body_evidence
            and shape_features["aspect"] < 1.28
        ):
            if pre_one_piece_score >= pre_top_score + 0.08:
                anchor = max(one_piece_score, bottom_score, pre_one_piece_score)
                refined[one_piece_index] = max(refined[one_piece_index], anchor * 1.05)
                refined[bottom_index] *= 0.55
                refined = refined / refined.sum()
                return refined, "topology_short_one_piece_upper_body"
            if pre_top_score >= pre_one_piece_score - 0.02:
                anchor = max(top_score, bottom_score, pre_top_score)
                refined[top_index] = max(refined[top_index], anchor * 1.05)
                refined[bottom_index] *= 0.48
                refined = refined / refined.sum()
                return refined, "topology_top_upper_body"

        if top_class in {"One_piece", "Top"} and bottom_score >= 0.10 and skirt_evidence and not upper_body_evidence:
            anchor = max(one_piece_score, top_score, bottom_score)
            refined[bottom_index] = max(refined[bottom_index], anchor * 1.12)
            refined[one_piece_index] *= 0.42
            refined[top_index] *= 0.42
            refined = refined / refined.sum()
            return refined, "topology_bottom_no_upper_body"

        if (
            top_class == "Bottom"
            and strong_upper_body_evidence
            and pre_one_piece_score >= pre_bottom_score + 0.10
        ):
            anchor = max(one_piece_score, bottom_score, pre_one_piece_score)
            refined[one_piece_index] = max(refined[one_piece_index], anchor * 1.12)
            refined[bottom_index] *= 0.42
            refined = refined / refined.sum()
            return refined, "topology_one_piece_strong_upper_body"

        if top_class == "Bottom" and upper_body_evidence and one_piece_score >= 0.22:
            boost = max(one_piece_score, bottom_score) * 0.30
            refined[one_piece_index] += boost
            refined[bottom_index] *= 0.82
            refined = refined / refined.sum()
            return refined, "topology_one_piece_upper_body"
        return None

    def _prediction_stats(self, probs: np.ndarray) -> tuple[str, float, float]:
        order = np.argsort(probs)[::-1]
        top_index = int(order[0])
        top_class = CLASS_NAMES[top_index]
        top_conf = float(probs[top_index])
        second_conf = float(probs[int(order[1])]) if len(order) > 1 else 0.0
        margin = top_conf - second_conf
        return top_class, top_conf, margin

    def _finalize_prediction(self, probs: np.ndarray) -> tuple[str, float, str]:
        class_name, confidence, _margin = self._prediction_stats(probs)

        if class_name == "Accessories" and confidence < 0.48:
            return "Unknown", confidence, "low_conf_accessories_unknown"
        if class_name in {"Shoes", "Bag"} and confidence < 0.42:
            return "Unknown", confidence, "low_conf_bag_shoes_unknown"
        if confidence < 0.38:
            return "Unknown", confidence, "low_conf_unknown"
        return class_name, confidence, ""

    def _top_scores(self, probs: np.ndarray, top_k: int = 4) -> Dict[str, float]:
        order = np.argsort(probs)[::-1][:top_k]
        return {CLASS_NAMES[int(index)]: float(probs[int(index)]) for index in order}

    def _shape_features(self, image: Image.Image) -> Dict[str, float] | None:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba, dtype=np.uint8)[..., 3] > 0
        ys, xs = np.where(alpha)
        if len(xs) == 0 or len(ys) == 0:
            return None

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        mask = alpha[y1:y2, x1:x2]
        box_h, box_w = mask.shape[:2]
        if box_h < 60 or box_w < 60:
            return None

        widths: List[float] = []
        top_segments: List[int] = []
        for row in mask:
            row_xs = np.where(row)[0]
            if len(row_xs) == 0:
                widths.append(0.0)
                continue
            widths.append(float(row_xs.max() - row_xs.min() + 1))
        widths_np = np.asarray(widths, dtype=np.float32)
        occupied = np.where(widths_np > 0)[0]
        if len(occupied) < 20:
            return None
        widths_np = widths_np[occupied[0]:occupied[-1] + 1]
        cropped_mask = mask[occupied[0]:occupied[-1] + 1]
        h = len(widths_np)
        max_width = float(widths_np.max())
        if max_width <= 1.0:
            return None

        def band(start_ratio: float, end_ratio: float, values: np.ndarray) -> float:
            start = max(0, int(h * start_ratio))
            end = max(start + 1, int(h * end_ratio))
            return float(np.median(values[start:end]))

        top_rows = cropped_mask[:max(1, int(h * 0.08))]
        for row in top_rows:
            row_uint = row.astype(np.uint8)
            segments = int(np.sum((row_uint[1:] == 1) & (row_uint[:-1] == 0)))
            if row_uint[0]:
                segments += 1
            top_segments.append(segments)

        center_slice = slice(int(box_w * 0.35), max(int(box_w * 0.65), int(box_w * 0.35) + 1))
        top_center_gap = 1.0 - float(np.mean(top_rows[:, center_slice]))

        return {
            "aspect": float(h) / float(max(1, box_w)),
            "top_ratio": band(0.0, 0.06, widths_np) / max_width,
            "upper_ratio": band(0.08, 0.18, widths_np) / max_width,
            "mid_ratio": band(0.35, 0.55, widths_np) / max_width,
            "lower_ratio": band(0.72, 0.92, widths_np) / max_width,
            "top_segments": float(np.mean(top_segments)) if top_segments else 0.0,
            "top_center_gap": top_center_gap,
        }


@lru_cache(maxsize=1)
def get_default_classifier(
    *,
    logger: Any | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> FlatlayItemClassifier:
    return FlatlayItemClassifier(logger=logger, device=device, batch_size=batch_size)
