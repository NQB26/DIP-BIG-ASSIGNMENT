"""Các hàm hình học thuần (không phụ thuộc OpenCV) để dễ unit-test."""
from __future__ import annotations

from typing import Sequence, Tuple

Point = Tuple[float, float]


def bbox_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def compute_ioa(inner: Sequence[float], outer: Sequence[float]) -> float:
    """IoA = Area(inner ∩ outer) / Area(inner).

    R1: inner = bbox lab_coat, outer = bbox person.
    """
    a = bbox_area(inner)
    return intersection_area(inner, outer) / a if a > 0 else 0.0


def bottom_center(b: Sequence[float]) -> Point:
    return ((b[0] + b[2]) / 2.0, b[3])


def center(b: Sequence[float]) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def point_in_bbox(p: Point, b: Sequence[float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def upper_body_roi(person: Sequence[float], ratio: float):
    """Vùng thân trên: từ y1 đến y1 + ratio * chiều cao."""
    return (person[0], person[1], person[2], person[1] + ratio * (person[3] - person[1]))


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting. Điểm nằm trên cạnh được tính là bên trong."""
    x, y = p
    n = len(poly)
    if n < 3:
        return False
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside
