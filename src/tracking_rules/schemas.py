"""Kiểu dữ liệu dùng chung giữa Detection -> Tracking -> Rule Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class Zone:
    zone_id: str
    type: str                                  # "general" | "restricted"
    polygon_norm: List[Tuple[float, float]]    # toạ độ chuẩn hoá [0, 1]

    def polygon_px(self, width: int, height: int):
        return [(x * width, y * height) for x, y in self.polygon_norm]
