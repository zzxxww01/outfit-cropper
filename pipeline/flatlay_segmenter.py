from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import clamp_bbox_xyxy, composite_on_white, crop_rgb, expand_bbox_xyxy, mask_to_bbox_xyxy


BBoxXYXY = Tuple[int, int, int, int]


@dataclass
class SegmentedItem:
    item_id: str
    bbox_xyxy: BBoxXYXY
    tile_bbox_xyxy: BBoxXYXY
    area_ratio: float
    mask: np.ndarray
    rgba_crop: Image.Image
    white_crop: Image.Image
    tile_crop: Image.Image
    preview_crop: Image.Image

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.bbox_xyxy
        tx1, ty1, tx2, ty2 = self.tile_bbox_xyxy
        return {
            "item_id": self.item_id,
            "bbox_xyxy": [x1, y1, x2, y2],
            "tile_bbox_xyxy": [tx1, ty1, tx2, ty2],
            "area_ratio": round(self.area_ratio, 6),
        }


@dataclass
class FlatlaySegmentationResult:
    items: List[SegmentedItem]
    relayout_image: Image.Image
    relayout_boxes: List[Dict[str, Any]]
    background_rgb: List[int]
    loose_threshold: float
    strong_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "background_rgb": self.background_rgb,
            "loose_threshold": round(self.loose_threshold, 3),
            "strong_threshold": round(self.strong_threshold, 3),
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "relayout_boxes": self.relayout_boxes,
        }


def segment_flatlay_image(
    image: Image.Image,
    *,
    min_component_area_ratio: float = 0.00008,
    strong_component_area_ratio: float = 0.0002,
    pad_ratio: float = 0.04,
    max_items: int = 8,
    relayout_gap_px: int = 56,
    relayout_margin_px: int = 72,
) -> FlatlaySegmentationResult:
    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_np.shape[:2]
    area_total = float(width * height)

    background_rgb = estimate_background(image_np)
    loose_threshold, strong_threshold = estimate_thresholds(image_np, background_rgb)
    loose_mask, strong_mask = build_foreground_masks(image_np, background_rgb, loose_threshold, strong_threshold)

    candidate_boxes = extract_candidate_boxes(
        loose_mask=loose_mask,
        width=width,
        height=height,
        area_total=area_total,
        min_component_area_ratio=min_component_area_ratio,
        max_items=max_items,
    )

    items: List[SegmentedItem] = []
    for index, box in enumerate(candidate_boxes):
        tile_bbox_xyxy = compute_safe_tile_bbox(
            box,
            candidate_boxes,
            width=width,
            height=height,
        )
        item = refine_item_mask(
            image_np=image_np,
            loose_mask=loose_mask,
            strong_mask=strong_mask,
            bbox_xyxy=box,
            tile_bbox_xyxy=tile_bbox_xyxy,
            item_id=f"item_{index}",
            area_total=area_total,
            pad_ratio=pad_ratio,
            min_component_area_ratio=strong_component_area_ratio,
        )
        if item is None:
            continue
        items.append(item)

    items = sorted(items, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))
    items = filter_final_items(items)
    items = merge_small_fragments(items, image_np=image_np, area_total=area_total)
    for index, item in enumerate(items):
        item.item_id = f"item_{index}"
    relayout_image, relayout_boxes = relayout_items(
        items,
        gap_px=relayout_gap_px,
        margin_px=relayout_margin_px,
        target_width=max(width + 220, 1800),
    )

    return FlatlaySegmentationResult(
        items=items,
        relayout_image=relayout_image,
        relayout_boxes=relayout_boxes,
        background_rgb=[int(v) for v in background_rgb.tolist()],
        loose_threshold=loose_threshold,
        strong_threshold=strong_threshold,
    )


def estimate_background(image_np: np.ndarray) -> np.ndarray:
    border = max(8, int(min(image_np.shape[:2]) * 0.02))
    top = image_np[:border, :, :].reshape(-1, 3)
    bottom = image_np[-border:, :, :].reshape(-1, 3)
    left = image_np[:, :border, :].reshape(-1, 3)
    right = image_np[:, -border:, :].reshape(-1, 3)
    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(border_pixels, axis=0)


def estimate_thresholds(image_np: np.ndarray, background_rgb: np.ndarray) -> tuple[float, float]:
    border = max(8, int(min(image_np.shape[:2]) * 0.02))
    border_pixels = np.concatenate(
        [
            image_np[:border, :, :].reshape(-1, 3),
            image_np[-border:, :, :].reshape(-1, 3),
            image_np[:, :border, :].reshape(-1, 3),
            image_np[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = background_rgb.astype(np.float32)
    border_dist = np.sqrt(np.sum((border_pixels - bg) ** 2, axis=1))
    loose_threshold = min(max(9.0, float(np.percentile(border_dist, 90.0)) + 4.0), 24.0)
    strong_threshold = min(max(16.0, loose_threshold + 7.0), 32.0)
    return loose_threshold, strong_threshold


def build_foreground_masks(
    image_np: np.ndarray,
    background_rgb: np.ndarray,
    loose_threshold: float,
    strong_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(np.uint8([[background_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.sqrt(np.sum((image_lab - background_lab) ** 2, axis=2))

    loose_mask = dist > loose_threshold
    strong_mask = dist > strong_threshold

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    loose_mask_u8 = (loose_mask.astype(np.uint8) * 255)
    strong_mask_u8 = (strong_mask.astype(np.uint8) * 255)

    loose_mask_u8 = cv2.morphologyEx(loose_mask_u8, cv2.MORPH_CLOSE, kernel_small)
    loose_mask_u8 = cv2.morphologyEx(loose_mask_u8, cv2.MORPH_OPEN, kernel_small)
    strong_mask_u8 = cv2.morphologyEx(strong_mask_u8, cv2.MORPH_CLOSE, kernel_medium)

    return loose_mask_u8 > 0, strong_mask_u8 > 0


def extract_candidate_boxes(
    *,
    loose_mask: np.ndarray,
    width: int,
    height: int,
    area_total: float,
    min_component_area_ratio: float,
    max_items: int,
) -> List[BBoxXYXY]:
    min_component_area = max(180, int(area_total * min_component_area_ratio))
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(loose_mask.astype(np.uint8), 8)

    boxes: List[tuple[int, BBoxXYXY]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        if x <= 1 or y <= 1 or x + bw >= width - 1 or y + bh >= height - 1:
            continue
        box = clamp_bbox_xyxy((x, y, x + bw, y + bh), width, height)
        boxes.append((area, box))

    boxes.sort(key=lambda item: item[0], reverse=True)
    filtered_boxes: List[BBoxXYXY] = []
    for _area, box in boxes:
        if is_redundant_box(box, filtered_boxes):
            continue
        filtered_boxes.append(box)
        if len(filtered_boxes) >= max_items:
            break
    return filtered_boxes


def refine_item_mask(
    *,
    image_np: np.ndarray,
    loose_mask: np.ndarray,
    strong_mask: np.ndarray,
    bbox_xyxy: BBoxXYXY,
    tile_bbox_xyxy: BBoxXYXY,
    item_id: str,
    area_total: float,
    pad_ratio: float,
    min_component_area_ratio: float,
) -> SegmentedItem | None:
    height, width = image_np.shape[:2]
    padded_box = expand_bbox_xyxy(bbox_xyxy, width, height, pad_ratio=pad_ratio)
    x1, y1, x2, y2 = padded_box
    original_box_area = max(1, (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))

    crop_rgb_np = image_np[y1:y2, x1:x2]
    crop_loose = loose_mask[y1:y2, x1:x2]
    crop_strong = strong_mask[y1:y2, x1:x2]
    if crop_rgb_np.size == 0:
        return None

    probable_fg = crop_loose.copy()
    if original_box_area >= 120000:
        probable_fg = build_support_mask(crop_loose)

    gc_mask = np.full(crop_loose.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_fg] = cv2.GC_PR_FGD
    gc_mask[crop_strong] = cv2.GC_FGD

    border = max(3, int(min(crop_loose.shape[:2]) * 0.03))
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_rgb_np, gc_mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
        crop_final = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    except cv2.error:
        crop_final = probable_fg.copy()

    crop_final_u8 = (crop_final.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    crop_final_u8 = cv2.morphologyEx(crop_final_u8, cv2.MORPH_OPEN, kernel)
    crop_final_u8 = cv2.morphologyEx(crop_final_u8, cv2.MORPH_CLOSE, kernel)
    crop_final = crop_final_u8 > 0

    if crop_final.sum() == 0:
        crop_final = probable_fg.copy()
    if crop_final.sum() == 0:
        return None

    min_component_area = max(60, int(area_total * min_component_area_ratio))
    crop_final = normalize_local_mask(crop_final, min_component_area=min_component_area)
    if should_try_rect_fallback(crop_final, original_box_area=original_box_area):
        rect_candidate = run_rect_grabcut(crop_rgb_np)
        rect_candidate = normalize_local_mask(rect_candidate, min_component_area=min_component_area)
        if prefer_rect_candidate(crop_final, rect_candidate):
            crop_final = rect_candidate

    local_bbox = mask_to_bbox_xyxy(crop_final.astype(np.uint8) * 255)
    if local_bbox is None:
        return None

    local_x1, local_y1, local_x2, local_y2 = local_bbox
    global_bbox = clamp_bbox_xyxy((x1 + local_x1, y1 + local_y1, x1 + local_x2, y1 + local_y2), width, height)
    gx1, gy1, gx2, gy2 = global_bbox

    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_mask[y1:y2, x1:x2][crop_final] = 255
    item_mask = final_mask[gy1:gy2, gx1:gx2]
    item_rgb = image_np[gy1:gy2, gx1:gx2]
    tx1, ty1, tx2, ty2 = tile_bbox_xyxy
    tile_rgb = image_np[ty1:ty2, tx1:tx2]
    alpha = item_mask.copy()

    rgba = np.dstack([item_rgb, alpha])
    rgba_image = Image.fromarray(rgba, mode="RGBA")
    white_crop = Image.fromarray(composite_on_white(item_rgb, item_mask))
    tile_crop = Image.fromarray(tile_rgb)
    preview_crop = white_crop
    if should_use_tile_preview(
        global_bbox=global_bbox,
        tile_bbox_xyxy=tile_bbox_xyxy,
        original_box_area=original_box_area,
    ):
        preview_crop = tile_crop
        white_crop = tile_crop

    area_ratio = float((item_mask > 0).sum()) / area_total
    return SegmentedItem(
        item_id=item_id,
        bbox_xyxy=global_bbox,
        tile_bbox_xyxy=tile_bbox_xyxy,
        area_ratio=area_ratio,
        mask=item_mask,
        rgba_crop=rgba_image,
        white_crop=white_crop,
        tile_crop=tile_crop,
        preview_crop=preview_crop,
    )


def build_support_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    kernel_size = max(9, min(25, (min(mask.shape[:2]) // 18) | 1))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return fill_mask_holes(closed > 0)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    mask_u8 = mask.astype(np.uint8) * 255
    height, width = mask_u8.shape[:2]
    flood = mask_u8.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def normalize_local_mask(mask: np.ndarray, *, min_component_area: int) -> np.ndarray:
    mask = keep_significant_local_components(mask, min_component_area=min_component_area)
    if not mask.any():
        return mask

    mask = keep_components_near_largest(mask, min_component_area=min_component_area)
    if not mask.any():
        return mask

    bbox = mask_to_bbox_xyxy(mask.astype(np.uint8) * 255)
    if bbox is None:
        return mask

    x1, y1, x2, y2 = bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    occupancy = float(mask.sum()) / float(bbox_area)
    if bbox_area >= 120000 and occupancy < 0.22:
        mask = fill_mask_holes(mask)
    return mask


def keep_components_near_largest(mask: np.ndarray, *, min_component_area: int) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 2:
        return mask

    components: List[tuple[int, int, int, int, int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append((label, area, x, y, w, h))

    if len(components) <= 1:
        return mask

    components.sort(key=lambda item: item[1], reverse=True)
    main_label, main_area, mx, my, mw, mh = components[0]
    main_box = (mx, my, mx + mw, my + mh)
    keep_labels = {main_label}

    for label, area, x, y, w, h in components[1:]:
        box = (x, y, x + w, y + h)
        gap = box_gap_simple(main_box, box)
        if area >= max(min_component_area * 2, int(main_area * 0.035)) or gap <= 42:
            keep_labels.add(label)

    keep = np.zeros(mask.shape, dtype=bool)
    for label in keep_labels:
        keep |= labels == label
    return keep


def run_rect_grabcut(crop_rgb_np: np.ndarray) -> np.ndarray:
    height, width = crop_rgb_np.shape[:2]
    border = max(3, int(min(height, width) * 0.03))
    rect_width = max(1, width - border * 2)
    rect_height = max(1, height - border * 2)
    rect = (border, border, rect_width, rect_height)

    gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_rgb_np, gc_mask, rect, bg_model, fg_model, 2, cv2.GC_INIT_WITH_RECT)
        return np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    except cv2.error:
        return np.zeros((height, width), dtype=bool)


def should_try_rect_fallback(mask: np.ndarray, *, original_box_area: int) -> bool:
    if original_box_area < 120000:
        return False
    area, bbox_area = local_mask_metrics(mask)
    if area == 0 or bbox_area == 0:
        return False
    occupancy = float(area) / float(bbox_area)
    return occupancy < 0.3


def prefer_rect_candidate(current_mask: np.ndarray, rect_mask: np.ndarray) -> bool:
    current_area, current_bbox_area = local_mask_metrics(current_mask)
    rect_area, rect_bbox_area = local_mask_metrics(rect_mask)
    if current_area == 0 or rect_area == 0 or rect_bbox_area == 0:
        return False

    current_occupancy = float(current_area) / float(current_bbox_area)
    rect_occupancy = float(rect_area) / float(rect_bbox_area)
    if rect_occupancy >= 0.9:
        return False
    return rect_area > current_area * 1.18 and rect_occupancy > current_occupancy + 0.08


def local_mask_metrics(mask: np.ndarray) -> tuple[int, int]:
    area = int(mask.sum())
    bbox = mask_to_bbox_xyxy(mask.astype(np.uint8) * 255)
    if bbox is None:
        return area, 0
    x1, y1, x2, y2 = bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return area, bbox_area


def should_use_tile_preview(
    *,
    global_bbox: BBoxXYXY,
    tile_bbox_xyxy: BBoxXYXY,
    original_box_area: int,
) -> bool:
    if original_box_area < 120000:
        return False

    gx1, gy1, gx2, gy2 = global_bbox
    tx1, ty1, tx2, ty2 = tile_bbox_xyxy
    global_bbox_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    tile_bbox_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    return tile_bbox_area > global_bbox_area * 1.6


def keep_significant_local_components(mask: np.ndarray, *, min_component_area: int) -> np.ndarray:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        keep |= labels == label
    if keep.any():
        return keep
    return mask


def relayout_items(
    items: List[SegmentedItem],
    *,
    gap_px: int,
    margin_px: int,
    target_width: int,
) -> tuple[Image.Image, List[Dict[str, Any]]]:
    if not items:
        return Image.new("RGB", (target_width, margin_px * 2 + 240), (245, 245, 245)), []

    placements: List[Dict[str, Any]] = []
    x = margin_px
    y = margin_px
    row_height = 0
    canvas_height = margin_px

    for item in items:
        item_width, item_height = item.preview_crop.size
        if x + item_width + margin_px > target_width:
            x = margin_px
            y += row_height + gap_px
            row_height = 0

        placements.append(
            {
                "item_id": item.item_id,
                "x": x,
                "y": y,
                "width": item_width,
                "height": item_height,
            }
        )
        x += item_width + gap_px
        row_height = max(row_height, item_height)
        canvas_height = max(canvas_height, y + item_height + margin_px)

    canvas = Image.new("RGB", (target_width, canvas_height), (245, 245, 245))
    for item, placement in zip(items, placements):
        canvas.paste(item.preview_crop.convert("RGB"), (placement["x"], placement["y"]))

    return canvas, placements


def filter_final_items(items: List[SegmentedItem]) -> List[SegmentedItem]:
    filtered: List[SegmentedItem] = []
    for item in items:
        width, height = item.tile_crop.size
        longest_side = max(width, height)
        if item.area_ratio < 0.0008 and longest_side < 180:
            continue
        filtered.append(item)
    return filtered


def merge_small_fragments(
    items: List[SegmentedItem],
    *,
    image_np: np.ndarray,
    area_total: float,
) -> List[SegmentedItem]:
    if not items:
        return items

    major_items: List[SegmentedItem] = []
    small_items: List[SegmentedItem] = []
    for item in items:
        if item.area_ratio < 0.0012:
            small_items.append(item)
        else:
            major_items.append(item)

    if not major_items:
        return items

    leftovers: List[SegmentedItem] = []
    for fragment in small_items:
        target_index = find_merge_target(fragment, major_items)
        if target_index is None:
            leftovers.append(fragment)
            continue
        major_items[target_index] = merge_items(major_items[target_index], fragment, image_np=image_np, area_total=area_total)

    merged = major_items + leftovers
    merged.sort(key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))
    return merged


def find_merge_target(fragment: SegmentedItem, items: List[SegmentedItem]) -> int | None:
    fx1, fy1, fx2, fy2 = fragment.bbox_xyxy
    fcx = (fx1 + fx2) / 2.0
    fcy = (fy1 + fy2) / 2.0

    best_index: int | None = None
    best_gap = 999999.0

    for index, item in enumerate(items):
        x1, y1, x2, y2 = item.bbox_xyxy
        gap = box_gap_simple(fragment.bbox_xyxy, item.bbox_xyxy)
        expanded = expand_bbox_xyxy(item.bbox_xyxy, width=100000, height=100000, pad_ratio=0.12)
        ex1, ey1, ex2, ey2 = expanded
        center_inside = ex1 <= fcx <= ex2 and ey1 <= fcy <= ey2
        axis_overlap = min(fy2, y2) - max(fy1, y1) > -24 or min(fx2, x2) - max(fx1, x1) > -24
        if center_inside or (gap <= 48 and axis_overlap):
            if gap < best_gap:
                best_gap = gap
                best_index = index
    return best_index


def merge_items(
    main_item: SegmentedItem,
    fragment_item: SegmentedItem,
    *,
    image_np: np.ndarray,
    area_total: float,
) -> SegmentedItem:
    mx1, my1, mx2, my2 = main_item.bbox_xyxy
    fx1, fy1, fx2, fy2 = fragment_item.bbox_xyxy
    ux1 = min(mx1, fx1)
    uy1 = min(my1, fy1)
    ux2 = max(mx2, fx2)
    uy2 = max(my2, fy2)

    union_mask = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.uint8)
    union_mask[my1 - uy1:my2 - uy1, mx1 - ux1:mx2 - ux1][main_item.mask > 0] = 255
    union_mask[fy1 - uy1:fy2 - uy1, fx1 - ux1:fx2 - ux1][fragment_item.mask > 0] = 255

    union_rgb = image_np[uy1:uy2, ux1:ux2]
    rgba = np.dstack([union_rgb, union_mask])
    rgba_image = Image.fromarray(rgba, mode="RGBA")
    white_crop = Image.fromarray(composite_on_white(union_rgb, union_mask))

    tx1, ty1, tx2, ty2 = main_item.tile_bbox_xyxy
    ftx1, fty1, ftx2, fty2 = fragment_item.tile_bbox_xyxy
    tile_bbox = (min(tx1, ftx1), min(ty1, fty1), max(tx2, ftx2), max(ty2, fty2))
    tile_rgb = image_np[tile_bbox[1]:tile_bbox[3], tile_bbox[0]:tile_bbox[2]]
    tile_crop = Image.fromarray(tile_rgb)
    preview_crop = white_crop
    original_box_area = max(1, (mx2 - mx1) * (my2 - my1))
    if should_use_tile_preview(
        global_bbox=(ux1, uy1, ux2, uy2),
        tile_bbox_xyxy=tile_bbox,
        original_box_area=original_box_area,
    ):
        preview_crop = tile_crop
        white_crop = tile_crop

    merged_area_ratio = float((union_mask > 0).sum()) / area_total
    return SegmentedItem(
        item_id=main_item.item_id,
        bbox_xyxy=(ux1, uy1, ux2, uy2),
        tile_bbox_xyxy=tile_bbox,
        area_ratio=merged_area_ratio,
        mask=union_mask,
        rgba_crop=rgba_image,
        white_crop=white_crop,
        tile_crop=tile_crop,
        preview_crop=preview_crop,
    )


def box_gap_simple(first: BBoxXYXY, second: BBoxXYXY) -> float:
    x1, y1, x2, y2 = first
    ox1, oy1, ox2, oy2 = second
    dx = max(x1 - ox2, ox1 - x2, 0)
    dy = max(y1 - oy2, oy1 - y2, 0)
    if dx == 0 and dy == 0:
        return 0.0
    if dx == 0:
        return float(dy)
    if dy == 0:
        return float(dx)
    return float((dx ** 2 + dy ** 2) ** 0.5)


def is_redundant_box(box: BBoxXYXY, kept_boxes: List[BBoxXYXY]) -> bool:
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for kept in kept_boxes:
        kx1, ky1, kx2, ky2 = kept
        inter_x1 = max(x1, kx1)
        inter_y1 = max(y1, ky1)
        inter_x2 = min(x2, kx2)
        inter_y2 = min(y2, ky2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            continue

        kept_area = max(1, (kx2 - kx1) * (ky2 - ky1))
        containment = inter_area / float(box_area)
        iou = inter_area / float(box_area + kept_area - inter_area)
        center_inside = kx1 <= cx <= kx2 and ky1 <= cy <= ky2
        if containment >= 0.55 or iou >= 0.18 or (center_inside and box_area < kept_area * 0.4):
            return True
    return False


def compute_safe_tile_bbox(
    box: BBoxXYXY,
    all_boxes: List[BBoxXYXY],
    *,
    width: int,
    height: int,
    pad_ratio: float = 0.18,
    min_pad_px: int = 56,
) -> BBoxXYXY:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(int(round(box_w * pad_ratio)), min_pad_px)
    pad_y = max(int(round(box_h * pad_ratio)), min_pad_px)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)

    for other in all_boxes:
        if other == box:
            continue
        ox1, oy1, ox2, oy2 = other
        vertical_overlap = min(y2, oy2) - max(y1, oy1)
        horizontal_overlap = min(x2, ox2) - max(x1, ox1)
        vertical_near = vertical_overlap > -max(24, int(min(box_h, oy2 - oy1) * 0.2))
        horizontal_near = horizontal_overlap > -max(24, int(min(box_w, ox2 - ox1) * 0.2))

        if ox1 >= x2 and vertical_near:
            right = min(right, (x2 + ox1) // 2)
        if ox2 <= x1 and vertical_near:
            left = max(left, (ox2 + x1) // 2)
        if oy1 >= y2 and horizontal_near:
            bottom = min(bottom, (y2 + oy1) // 2)
        if oy2 <= y1 and horizontal_near:
            top = max(top, (oy2 + y1) // 2)

    return clamp_bbox_xyxy((left, top, right, bottom), width, height)
