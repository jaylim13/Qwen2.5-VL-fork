#!/usr/bin/env python3
"""
Convert clip-level JSON to frame-range JSON format.
Produces two output files:
  - one using predicted_label
  - one using model_prediction
Assumes 30 FPS and 2-second clips (60 frames per clip).
"""

import json
from pathlib import Path

# ==========================
# CONFIG
# ==========================
INPUT_FILE = Path("predicted_vs_actual_2/video_4_hybrid.json")
OUTPUT_PREDICTED = Path("frames_predicted_label_4.json")
OUTPUT_MODEL     = Path("frames_model_prediction_4.json")

FPS             = 30
FRAMES_PER_CLIP = FPS * 2  # 60 frames per 2-second clip

# Map from action_label values → output JSON keys
LABEL_MAP = {
    "sitting":      "Sitting",
    "standing":     "Standing still",
    "walking":      "Walking",
    "go upstair":   "Upstair",
    "go downstair": "Downstair",
}

# ==========================
# CONVERSION FUNCTION
# ==========================
def clips_to_frames(data, field):
    """
    Convert clip list to frame ranges grouped by action label.
    Merges consecutive clips with the same label into a single range.
    """
    # Sort by clip index
    data = sorted(data, key=lambda x: x["clip_index"])

    # Build (label, start_frame, end_frame) per clip
    clip_ranges = []
    for i, d in enumerate(data):
        label = str(d.get(field, "") or "").strip().lower()
        label = LABEL_MAP.get(label, label)
        start_frame = i * FRAMES_PER_CLIP + 1
        end_frame   = (i + 1) * FRAMES_PER_CLIP
        clip_ranges.append((label, start_frame, end_frame))

    # Merge consecutive clips with same label
    merged = []
    for label, start, end in clip_ranges:
        if merged and merged[-1]["label"] == label and merged[-1]["end"] == start - 1:
            merged[-1]["end"] = end
        else:
            merged.append({"label": label, "start": start, "end": end})

    # Group into output format
    all_labels = list(LABEL_MAP.values())
    result = {label: [] for label in all_labels}

    for entry in merged:
        label = entry["label"]
        if label not in result:
            result[label] = []
        result[label].append({
            "Starting frame": entry["start"],
            "Ending frame":   entry["end"]
        })

    return result

# ==========================
# MAIN
# ==========================
with open(INPUT_FILE) as f:
    data = json.load(f)

# predicted_label output
predicted_frames = clips_to_frames(data, "predicted_label")
with open(OUTPUT_PREDICTED, "w") as f:
    json.dump(predicted_frames, f, indent=2)
print(f"✅ Saved: {OUTPUT_PREDICTED}")

# model_prediction output
model_frames = clips_to_frames(data, "model_prediction")
with open(OUTPUT_MODEL, "w") as f:
    json.dump(model_frames, f, indent=2)
print(f"✅ Saved: {OUTPUT_MODEL}")