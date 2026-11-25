"""Single place that knows how to talk to the tracker.

TV2's ``tracking_rules.PersonTracker`` (ByteTrack) was not part of the files
handed to TV3, so its exact constructor/``update`` signature is discovered with
``inspect`` instead of being hard-coded. If the real tracker's signature does
not map, fix ONLY this file (``PersonTrackerAdapter``).

``SimpleIoUTracker`` is a clearly-labelled fallback (``--tracker simple``) for
smoke tests; it is NOT ByteTrack and should not be used for reported metrics.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Sequence

from tracking_rules.schemas import Detection, Track

_FPS_NAMES = ("fps", "frame_rate", "video_fps")
_FRAME_ID_NAMES = ("frame_id", "frame_idx", "frame_index", "idx", "frame_no", "frame_number")
_FRAME_NAMES = ("frame", "image", "img")
_TIME_NAMES = ("t", "timestamp", "time_sec", "video_time_sec", "time")
_SIZE_NAMES = ("frame_size", "size", "shape")
_DET_NAMES = ("detections", "dets", "detection_list")


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_tracks(obj: Any) -> List[Track]:
    """Normalise whatever the tracker returns into a list of Track."""
    if obj is None:
        return []
    out: List[Track] = []
    for item in obj:
        if isinstance(item, Track):
            out.append(item)
        elif isinstance(item, dict):
            cls = item.get("class_name", item.get("class", "person"))
            out.append(Track(int(item["track_id"]), str(cls),
                             float(item.get("confidence", item.get("conf", 1.0))),
                             tuple(float(v) for v in item["bbox"])))
        else:
            raise TypeError(f"Unsupported tracker output item: {type(item).__name__}")
    return out


class PersonTrackerAdapter:
    """Wraps tracking_rules.PersonTracker without assuming its signature."""

    def __init__(self, tracker_cls, tracker_cfg: dict, fps: float):
        sig = inspect.signature(tracker_cls.__init__)
        params = {n: p for n, p in sig.parameters.items() if n != "self"}
        kwargs: Dict[str, Any] = {}
        for name in params:
            if name in _FPS_NAMES:
                kwargs[name] = fps
            elif name in tracker_cfg:
                kwargs[name] = tracker_cfg[name]
            elif name in ("cfg", "config", "tracker_cfg", "tracker_config"):
                kwargs[name] = tracker_cfg
        self.tracker = tracker_cls(**kwargs)
        self._used_ctor_kwargs = sorted(kwargs)
        method = getattr(self.tracker, "update", None) or self.tracker
        self._call = method
        self._update_params = [n for n in inspect.signature(method).parameters]

    def update(self, detections: Sequence[Detection], frame_id: int, t: float,
               frame, frame_size) -> List[Track]:
        kwargs: Dict[str, Any] = {}
        positional_fallback = True
        for n in self._update_params:
            if n in _DET_NAMES:
                kwargs[n] = detections
            elif n in _FRAME_ID_NAMES:
                kwargs[n] = frame_id
            elif n in _FRAME_NAMES:
                kwargs[n] = frame
            elif n in _TIME_NAMES:
                kwargs[n] = t
            elif n in _SIZE_NAMES:
                kwargs[n] = frame_size
        if any(n in kwargs for n in _DET_NAMES):
            positional_fallback = False
        if positional_fallback:
            return to_tracks(self._call(detections, frame_id))
        return to_tracks(self._call(**kwargs))


class SimpleIoUTracker:
    """Greedy IoU tracker for person boxes (fallback / tests only)."""

    def __init__(self, min_iou: float = 0.3, max_age_frames: int = 15, min_conf: float = 0.0):
        self.min_iou = min_iou
        self.max_age = max_age_frames
        self.min_conf = min_conf
        self._next = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, detections, frame_id, t, frame, frame_size) -> List[Track]:
        persons = [d for d in detections if d.class_name == "person" and d.confidence >= self.min_conf]
        pairs = sorted(((_iou(d.bbox, tr["bbox"]), di, tid)
                        for di, d in enumerate(persons) for tid, tr in self._tracks.items()),
                       reverse=True)
        used_d, used_t, out = set(), set(), []
        for score, di, tid in pairs:
            if score < self.min_iou or di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            self._tracks[tid].update(bbox=persons[di].bbox, last=frame_id)
            out.append(Track(tid, "person", persons[di].confidence, persons[di].bbox))
        for di, d in enumerate(persons):
            if di in used_d:
                continue
            tid, self._next = self._next, self._next + 1
            self._tracks[tid] = {"bbox": d.bbox, "last": frame_id}
            out.append(Track(tid, "person", d.confidence, d.bbox))
        for tid in [k for k, v in self._tracks.items() if frame_id - v["last"] > self.max_age]:
            del self._tracks[tid]
        return out


def build_tracker(tracker_cfg: dict, fps: float, kind: str = "auto"):
    """kind: auto (PersonTracker, else error) | bytetrack | simple."""
    if kind == "simple":
        return SimpleIoUTracker()
    try:
        from tracking_rules import PersonTracker
    except ImportError as exc:
        raise RuntimeError(
            "tracking_rules.PersonTracker is not importable "
            f"({exc}). Check PYTHONPATH, or use --tracker simple for a smoke test."
        ) from exc
    return PersonTrackerAdapter(PersonTracker, tracker_cfg, fps)
