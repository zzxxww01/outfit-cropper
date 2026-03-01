from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class ItemTypeEnum(str, Enum):
    outerwear = "outerwear"
    top = "top"
    bottom = "bottom"
    one_piece = "one_piece"
    shoes = "shoes"
    bag = "bag"
    accessories = "accessories"
    unknown = "unknown"


class GpuMetaItem(BaseModel):
    item_image_path: str = Field(description="Relative path under outfit output folder.")
    bbox: List[float] = Field(description="[ymin, xmin, ymax, xmax] in 0-1000 coordinates.")
    is_fallback: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: List[float]) -> List[float]:
        if len(value) != 4:
            raise ValueError("bbox must contain exactly 4 numeric values.")
        ymin, xmin, ymax, xmax = value
        if not (0.0 <= ymin <= ymax <= 1000.0):
            raise ValueError("bbox y-axis must satisfy 0 <= ymin <= ymax <= 1000.")
        if not (0.0 <= xmin <= xmax <= 1000.0):
            raise ValueError("bbox x-axis must satisfy 0 <= xmin <= xmax <= 1000.")
        return [round(float(v), 2) for v in value]


class GpuMeta(BaseModel):
    outfit_id: str
    items: List[GpuMetaItem] = Field(default_factory=list)


class BatchErrorItem(BaseModel):
    outfit_id: str
    failed_step: str
    message: str
    retries: int = Field(default=0, ge=0)


class BatchErrorReport(BaseModel):
    total_images: int = Field(default=0, ge=0)
    failed_images: int = Field(default=0, ge=0)
    errors: List[BatchErrorItem] = Field(default_factory=list)

