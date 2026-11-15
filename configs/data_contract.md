```json
// 1. Đầu ra từ Tracking (truyền vào Rule Engine)
{
  "frame_id": 145,
  "timestamp": 4.83,
  "tracks": [
    {
      "track_id": 3,
      "class": "person",
      "confidence": 0.91,
      "bbox": [320, 140, 450, 480]
    }
  ],
  "objects": [
    {
      "class": "cell_phone",
      "confidence": 0.84,
      "bbox": [375, 240, 405, 285]
    }
  ]
}

// 2. Đầu ra từ Rule Engine (gửi sang Event Manager & Dashboard của TV3)
{
  "event_id": "EVT_20261115_0001",
  "rule_id": "R3",
  "rule_name": "PHONE_USAGE",
  "track_id": 3,
  "status": "CONFIRMED",
  "timestamp": "2026-11-15T10:15:32.400",
  "video_time_sec": 4.83,
  "duration_sec": 3.05,
  "zone_id": "lab_main_area",
  "bbox": [320, 140, 450, 480],
  "evidence": {
    "image_path": null,
    "clip_path": null
  }
}
```