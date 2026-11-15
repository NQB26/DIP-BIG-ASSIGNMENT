from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Detection:
    class_name: str          # VD: "person", "lab_coat", "cell_phone"
    confidence: float        # VD: 0.89
    bbox: List[float]        # [x1, y1, x2, y2]

@dataclass
class Track:
    track_id: int            # VD: 3
    class_name: str
    confidence: float
    bbox: List[float]

@dataclass
class RuleEvent:
    rule_id: str             # "R1", "R2", "R3"
    rule_name: str           # "NO_LAB_COAT", "RESTRICTED_ZONE", "PHONE_USAGE"
    track_id: int
    status: str              # "CONFIRMED"
    timestamp: float         # Thời điểm trong video (giây)
    bbox: List[float]
    evidence_path: Optional[str] = None