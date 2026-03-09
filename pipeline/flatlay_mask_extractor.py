from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from pipeline.flatlay_segmenter import (
    SegmentedItem,
    box_gap_simple,
    merge_items,
    relayout_items,
    segment_flatlay_image,
)


@dataclass
class MaskExtractedItem:
    item_id: str
    bbox_xyxy: tuple[int, int, int, int]
    tile_bbox_xyxy: tuple[int, int, int, int]
    area_ratio: float
    mask: np.ndarray
    rgba_crop: Image.Image
    white_crop: Image.Image
    tile_crop: Image.Image
    preview_crop: Image.Image
    source_method: str = "mask_seg_flatlay"
    group_type: str = "single"
    class_name: str = "Unknown"
    class_confidence: float = 0.0
    classification_stage: str = ""
    top_scores: Dict[str, float] | None = None
    stage2_triggered: bool = False
    topology_reranked: bool = False
    decision_reason: str = ""
    shape_features: Dict[str, float] | None = None
    merged_from: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.bbox_xyxy
        data: Dict[str, Any] = {
            "item_id": self.item_id,
            "bbox_xyxy": [x1, y1, x2, y2],
            "area_ratio": round(self.area_ratio, 6),
            "source_method": self.source_method,
            "group_type": self.group_type,
            "class_name": self.class_name,
            "class_confidence": round(self.class_confidence, 6),
        }
        if self.classification_stage:
            data["classification_stage"] = self.classification_stage
        if self.top_scores:
            data["top_scores"] = {name: round(score, 6) for name, score in self.top_scores.items()}
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
        if self.merged_from:
            data["merged_from"] = list(self.merged_from)
        return data


@dataclass
class MaskExtractionResult:
    items: List[MaskExtractedItem]
    relayout_image: Image.Image
    relayout_boxes: List[Dict[str, Any]]
    warnings: List[str]
    pipeline: str = "mask_seg_flatlay"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "item_count": len(self.items),
            "warnings": list(self.warnings),
            "items": [item.to_dict() for item in self.items],
            "relayout_boxes": self.relayout_boxes,
        }


class FlatlayMaskExtractor:
    def __init__(
        self,
        *,
        max_items: int = 10,
        logger: Any | None = None,
        min_component_area_ratio: float = 0.00008,
        strong_component_area_ratio: float = 0.0002,
        output_pad_ratio: float = 0.08,
        min_output_pad_px: int = 24,
    ) -> None:
        self.max_items = max_items
        self.logger = logger
        self.min_component_area_ratio = min_component_area_ratio
        self.strong_component_area_ratio = strong_component_area_ratio
        self.output_pad_ratio = output_pad_ratio
        self.min_output_pad_px = min_output_pad_px

    def extract_items(self, image: Image.Image) -> MaskExtractionResult:
        image_rgb = image.convert("RGB")
        image_np = np.asarray(image_rgb, dtype=np.uint8)
        image_height, image_width = image_np.shape[:2]
        area_total = float(image_width * image_height)

        segmentation = segment_flatlay_image(
            image_rgb,
            min_component_area_ratio=self.min_component_area_ratio,
            strong_component_area_ratio=self.strong_component_area_ratio,
            max_items=self.max_items,
        )
        items = [self._from_segmented_item(item) for item in segmentation.items]
        warnings: List[str] = []

        items = self._drop_micro_noise(items)
        items = self._merge_probable_pairs(
            items,
            image_np=image_np,
            area_total=area_total,
            image_height=image_height,
            pair_kind="shoe_pair",
        )
        items = self._merge_probable_pairs(
            items,
            image_np=image_np,
            area_total=area_total,
            image_height=image_height,
            pair_kind="accessory_pair",
        )
        items = [self._pad_item_canvas(item) for item in items]
        items = self._sort_and_renumber(items)

        if not items:
            warnings.append("no_items_extracted")

        relayout_image, relayout_boxes = relayout_items(
            [self._to_segmented_item(item) for item in items],
            gap_px=56,
            margin_px=72,
            target_width=max(image_width + 220, 1800),
        )
        return MaskExtractionResult(
            items=items,
            relayout_image=relayout_image,
            relayout_boxes=relayout_boxes,
            warnings=warnings,
        )

    def _drop_micro_noise(self, items: Sequence[MaskExtractedItem]) -> List[MaskExtractedItem]:
        if len(items) <= 2:
            return list(items)

        kept: List[MaskExtractedItem] = []
        for item in items:
            longest_side = max(item.rgba_crop.size)
            if item.area_ratio < 0.00055 and longest_side < 88:
                continue
            kept.append(item)
        return kept or list(items)

    def _merge_probable_pairs(
        self,
        items: Sequence[MaskExtractedItem],
        *,
        image_np: np.ndarray,
        area_total: float,
        image_height: int,
        pair_kind: str,
    ) -> List[MaskExtractedItem]:
        if len(items) < 2:
            return list(items)

        candidates: List[tuple[float, int, int]] = []
        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                score = self._pair_score(
                    items[left_index],
                    items[right_index],
                    image_height=image_height,
                    pair_kind=pair_kind,
                )
                if score is None:
                    continue
                candidates.append((score, left_index, right_index))

        candidates.sort(key=lambda item: item[0])
        used_indexes: set[int] = set()
        merged_items: List[MaskExtractedItem] = []
        for score, left_index, right_index in candidates:
            if left_index in used_indexes or right_index in used_indexes:
                continue
            left_item = items[left_index]
            right_item = items[right_index]
            merged_segment = merge_items(
                self._to_segmented_item(left_item),
                self._to_segmented_item(right_item),
                image_np=image_np,
                area_total=area_total,
            )
            merged_items.append(
                self._from_segmented_item(
                    merged_segment,
                    group_type=pair_kind,
                    merged_from=[left_item.item_id, right_item.item_id],
                )
            )
            used_indexes.add(left_index)
            used_indexes.add(right_index)
            if self.logger is not None:
                self.logger.debug(
                    "Merged %s from %s and %s score=%.4f",
                    pair_kind,
                    left_item.item_id,
                    right_item.item_id,
                    score,
                )

        leftovers = [item for index, item in enumerate(items) if index not in used_indexes]
        return leftovers + merged_items

    def _pair_score(
        self,
        left_item: MaskExtractedItem,
        right_item: MaskExtractedItem,
        *,
        image_height: int,
        pair_kind: str,
    ) -> float | None:
        lx1, ly1, lx2, ly2 = left_item.bbox_xyxy
        rx1, ry1, rx2, ry2 = right_item.bbox_xyxy
        left_width = lx2 - lx1
        left_height = ly2 - ly1
        right_width = rx2 - rx1
        right_height = ry2 - ry1
        left_cy = (ly1 + ly2) / 2.0
        right_cy = (ry1 + ry2) / 2.0

        size_ratio = max(left_item.area_ratio, right_item.area_ratio) / max(
            min(left_item.area_ratio, right_item.area_ratio),
            1e-6,
        )
        if size_ratio > 1.8:
            return None

        gap = box_gap_simple(left_item.bbox_xyxy, right_item.bbox_xyxy)
        similarity = self._visual_distance(left_item, right_item)
        if pair_kind == "shoe_pair":
            if left_item.area_ratio < 0.004 or right_item.area_ratio < 0.004:
                return None
            if left_item.area_ratio > 0.07 or right_item.area_ratio > 0.07:
                return None
            if max(left_width, left_height, right_width, right_height) < 90:
                return None
            if min(left_cy, right_cy) < image_height * 0.43:
                return None
            if abs(left_cy - right_cy) > max(left_height, right_height) * 0.6 + 48:
                return None
            if gap > max(left_width, right_width) * 1.15 + 68:
                return None
            if similarity > 0.18:
                return None
            return similarity + gap / 1800.0

        if pair_kind == "accessory_pair":
            if left_item.area_ratio > 0.006 or right_item.area_ratio > 0.006:
                return None
            if max(left_width, left_height, right_width, right_height) > 220:
                return None
            if abs(left_cy - right_cy) > max(left_height, right_height) * 0.55 + 24:
                return None
            if gap > max(left_width, right_width) * 1.8 + 40:
                return None
            if similarity > 0.08:
                return None
            if abs(left_cy - right_cy) > 48:
                return None
            return similarity + gap / 2200.0

        return None

    def _visual_distance(self, left_item: MaskExtractedItem, right_item: MaskExtractedItem) -> float:
        left_thumb = self._item_thumbnail(left_item)
        right_thumb = self._item_thumbnail(right_item)
        diff = np.mean(np.abs(left_thumb - right_thumb))
        flipped_diff = np.mean(np.abs(left_thumb - right_thumb[:, ::-1, :]))
        return float(min(diff, flipped_diff))

    def _item_thumbnail(self, item: MaskExtractedItem) -> np.ndarray:
        rgba = item.rgba_crop.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
        rgba_np = np.asarray(rgba, dtype=np.float32) / 255.0
        return rgba_np[..., :3] * rgba_np[..., 3:4]

    def _sort_and_renumber(self, items: Sequence[MaskExtractedItem]) -> List[MaskExtractedItem]:
        sorted_items = sorted(items, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))
        return [replace(item, item_id=f"item_{index}") for index, item in enumerate(sorted_items)]

    def _pad_item_canvas(self, item: MaskExtractedItem) -> MaskExtractedItem:
        width, height = item.rgba_crop.size
        pad_x = max(int(round(width * self.output_pad_ratio)), self.min_output_pad_px)
        pad_y = max(int(round(height * self.output_pad_ratio)), self.min_output_pad_px)
        new_size = (width + pad_x * 2, height + pad_y * 2)

        rgba_canvas = Image.new("RGBA", new_size, (255, 255, 255, 0))
        rgba_canvas.paste(item.rgba_crop, (pad_x, pad_y), item.rgba_crop)

        white_canvas = Image.new("RGB", new_size, (255, 255, 255))
        white_canvas.paste(item.white_crop.convert("RGB"), (pad_x, pad_y))

        preview_canvas = Image.new("RGB", new_size, (255, 255, 255))
        preview_canvas.paste(item.preview_crop.convert("RGB"), (pad_x, pad_y))

        tile_canvas = Image.new("RGB", new_size, (255, 255, 255))
        tile_canvas.paste(item.tile_crop.convert("RGB"), (pad_x, pad_y))

        padded_mask = np.zeros((new_size[1], new_size[0]), dtype=np.uint8)
        padded_mask[pad_y:pad_y + item.mask.shape[0], pad_x:pad_x + item.mask.shape[1]] = item.mask

        return replace(
            item,
            rgba_crop=rgba_canvas,
            white_crop=white_canvas,
            preview_crop=preview_canvas,
            tile_crop=tile_canvas,
            mask=padded_mask,
        )

    def _from_segmented_item(
        self,
        item: SegmentedItem,
        *,
        group_type: str = "single",
        merged_from: List[str] | None = None,
    ) -> MaskExtractedItem:
        return MaskExtractedItem(
            item_id=item.item_id,
            bbox_xyxy=item.bbox_xyxy,
            tile_bbox_xyxy=item.tile_bbox_xyxy,
            area_ratio=item.area_ratio,
            mask=item.mask,
            rgba_crop=item.rgba_crop,
            white_crop=item.white_crop,
            tile_crop=item.tile_crop,
            preview_crop=item.preview_crop,
            group_type=group_type,
            merged_from=merged_from,
        )

    def _to_segmented_item(self, item: MaskExtractedItem) -> SegmentedItem:
        return SegmentedItem(
            item_id=item.item_id,
            bbox_xyxy=item.bbox_xyxy,
            tile_bbox_xyxy=item.tile_bbox_xyxy,
            area_ratio=item.area_ratio,
            mask=item.mask,
            rgba_crop=item.rgba_crop,
            white_crop=item.white_crop,
            tile_crop=item.tile_crop,
            preview_crop=item.preview_crop,
        )
