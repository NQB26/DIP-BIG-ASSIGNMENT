"""FSM lọc thời gian cho mỗi cặp (track_id, rule_id).

NORMAL --vi phạm--> PENDING --đủ min_duration--> CONFIRMED (phát 1 event)
PENDING --mất vi phạm > grace--> NORMAL
CONFIRMED --(ngay bước kế)--> COOLDOWN --hết cooldown--> NORMAL

Thời gian là *video time* (frame_id / fps), không phải đồng hồ hệ thống,
nên chạy offline và real-time cho cùng kết quả.
"""
from __future__ import annotations

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


class ViolationFSM:
    def __init__(self, cfg: FSMConfig):
        self.cfg = cfg
        self.state = State.NORMAL
        self.start_t: Optional[float] = None
        self.last_violation_t: Optional[float] = None
        self.cooldown_until: float = 0.0
        self.last_update_t: float = 0.0

    @property
    def elapsed(self) -> float:
        """Thời gian vi phạm tích luỹ (cho overlay / debug)."""
        if self.state == State.PENDING and self.start_t is not None:
            return (self.last_violation_t or self.start_t) - self.start_t
        return 0.0

    def update(self, violated: bool, t: float) -> bool:
        """Cập nhật 1 bước. Trả về True đúng 1 lần tại thời điểm CONFIRMED."""
        self.last_update_t = t

        if self.state == State.CONFIRMED:           # trạng thái thoáng qua
            self.state = State.COOLDOWN
            self.cooldown_until = t + self.cfg.cooldown_sec

        if self.state == State.COOLDOWN:
            if t >= self.cooldown_until:
                self.state = State.NORMAL
                self.start_t = self.last_violation_t = None
            else:
                return False

        if self.state == State.NORMAL:
            if violated:
                self.state = State.PENDING
                self.start_t = self.last_violation_t = t
                return self._maybe_confirm(t)
            return False

        if self.state == State.PENDING:
            if violated:
                self.last_violation_t = t
                return self._maybe_confirm(t)
            if t - self.last_violation_t > self.cfg.grace_period_sec:
                self.state = State.NORMAL
                self.start_t = self.last_violation_t = None
        return False

    def _maybe_confirm(self, t: float) -> bool:
        if t - self.start_t >= self.cfg.min_duration_sec:
            self.state = State.CONFIRMED
            return True
        return False
