"""FSM lọc thời gian cho mỗi cặp (track_id, rule_id).

NORMAL -> PENDING -> CONFIRMED -> COOLDOWN -> NORMAL

Thời gian đầu vào là video time (giây), không phải wall-clock. Nếu một
khoảng trống quan sát dài hơn grace_period_sec xuất hiện, bộ đếm PENDING
được reset kể cả khi frame kế tiếp lại báo vi phạm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(str, Enum):
    NORMAL = "NORMAL"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class FSMConfig:
    min_duration_sec: float
    grace_period_sec: float = 0.4
    cooldown_sec: float = 10.0

    def __post_init__(self) -> None:
        for name, value in (
            ("min_duration_sec", self.min_duration_sec),
            ("grace_period_sec", self.grace_period_sec),
            ("cooldown_sec", self.cooldown_sec),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


class ViolationFSM:
    def __init__(self, cfg: FSMConfig):
        self.cfg = cfg
        self.state = State.NORMAL
        self.start_t: Optional[float] = None
        self.last_violation_t: Optional[float] = None
        self.cooldown_until: float = 0.0
        self.last_update_t: Optional[float] = None

    @property
    def elapsed(self) -> float:
        """Thời gian vi phạm đã được quan sát liên tục (cho overlay/debug)."""
        if self.state == State.PENDING and self.start_t is not None:
            last = self.last_violation_t
            return max(0.0, (last if last is not None else self.start_t) - self.start_t)
        return 0.0

    def update(self, violated: bool, t: float) -> bool:
        """Trả về True đúng một lần khi vi phạm đạt min_duration_sec."""
        t = float(t)
        if not math.isfinite(t) or t < 0:
            raise ValueError("video timestamp must be finite and non-negative")
        if self.last_update_t is not None and t < self.last_update_t:
            raise ValueError(
                f"video timestamps must be monotonic: {t} < {self.last_update_t}"
            )
        self.last_update_t = t

        if self.state == State.CONFIRMED:
            # cooldown_until đã được cố định tại thời điểm confirmation.
            self.state = State.COOLDOWN

        if self.state == State.COOLDOWN:
            if t < self.cooldown_until:
                return False
            self.state = State.NORMAL
            self.start_t = None
            self.last_violation_t = None

        if self.state == State.NORMAL:
            if not violated:
                return False
            self.state = State.PENDING
            self.start_t = t
            self.last_violation_t = t
            return self._maybe_confirm(t)

        # PENDING
        if violated:
            # Không được cộng dồn thời gian qua một khoảng mất quan sát dài.
            if (
                self.last_violation_t is not None
                and t - self.last_violation_t > self.cfg.grace_period_sec
            ):
                self.start_t = t
            self.last_violation_t = t
            return self._maybe_confirm(t)

        if (
            self.last_violation_t is not None
            and t - self.last_violation_t > self.cfg.grace_period_sec
        ):
            self.state = State.NORMAL
            self.start_t = None
            self.last_violation_t = None
        return False

    def _maybe_confirm(self, t: float) -> bool:
        if self.start_t is not None and t - self.start_t >= self.cfg.min_duration_sec:
            self.state = State.CONFIRMED
            self.cooldown_until = t + self.cfg.cooldown_sec
            return True
        return False
