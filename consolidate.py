#!/usr/bin/env python3
"""
Consolidate 3 per-video JSON files (hybrid, rule_based, zero_shot)
into a single merged JSON per video with all original fields plus:
  - hybrid_prediction
  - SLAM_prediction
  - zero_shot_prediction
  - model_prediction (out-of-box, from any file that has it)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ==========================
# CONFIG
# ==========================
INPUT_FOLDER  = Path("predicted_vs_actual_2")
OUTPUT_FOLDER = Path("predicted_vs_actual_consolidated")
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ==========================
# GROUP FILES BY VIDEO NAME
# ==========================
def find_video_files(folder):
    groups = defaultdict(dict)
    for p in sorted(folder.glob("*.json")):
        for suffix in ("hybrid", "rule_based", "zero_shot"):
            if p.stem.endswith(f"_{suffix}"):
                video_name = p.stem[: -(len(suffix) + 1)]
                groups[video_name][suffix] = p
                break
    return groups

# ==========================
# LOAD HELPERS
# ==========================
def load_standard(path):
    """Load hybrid/rule_based files — keyed by clip_index."""
    with open(path) as f:
        data = json.load(f)
    return {d["clip_index"]: d for d in data}

def load_zero_shot(path):
    """
    Load zero_shot file — keyed by clip_index extracted from video_path.
    e.g. 'data/video_j2/video_clips/clip0003.mp4' → 'clip_0003'
    """
    with open(path) as f:
        data = json.load(f)
    result = {}
    for d in data:
        match = re.search(r"clip(\d+)", d.get("video_path", ""))
        if match:
            clip_index = f"clip_{int(match.group(1)):04d}"
            result[clip_index] = d
    return result

# ==========================
# CONSOLIDATE
# ==========================
def consolidate(video_name, file_map):
    loaded_standard = {}
    loaded_zero_shot = {}

    for suffix, path in file_map.items():
        if suffix == "zero_shot":
            loaded_zero_shot = load_zero_shot(path)
        else:
            loaded_standard[suffix] = load_standard(path)

    # Use hybrid as base, fall back to rule_based
    base_suffix = "hybrid" if "hybrid" in loaded_standard else next(iter(loaded_standard))
    base_clips  = loaded_standard[base_suffix]
    clip_ids    = sorted(base_clips.keys())

    merged = []
    for clip_id in clip_ids:
        clip = dict(base_clips[clip_id])

        # Remove ambiguous predicted_label from base
        clip.pop("predicted_label", None)

        # hybrid_prediction: predicted_label from hybrid file
        clip["hybrid_prediction"] = (
            loaded_standard.get("hybrid", {}).get(clip_id, {}).get("predicted_label")
        )

        # SLAM_prediction: predicted_label from rule_based file
        clip["SLAM_prediction"] = (
            loaded_standard.get("rule_based", {}).get(clip_id, {}).get("predicted_label")
        )

        # zero_shot_prediction: classified_action from zero_shot file
        clip["zero_shot_prediction"] = (
            loaded_zero_shot.get(clip_id, {}).get("classified_action")
        )

        merged.append(clip)

    out_path = OUTPUT_FOLDER / f"{video_name}.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✅ {video_name}.json — {len(merged)} clips")
    missing_zs = sum(1 for c in merged if c["zero_shot_prediction"] is None)
    if missing_zs:
        print(f"   ⚠️  {missing_zs} clips missing zero_shot_prediction")

# ==========================
# MAIN
# ==========================
video_groups = find_video_files(INPUT_FOLDER)

if not video_groups:
    print(f"No matching files found in {INPUT_FOLDER}")
    print("Expected: video_1_hybrid.json, video_1_rule_based.json, video_1_zero_shot.json")
else:
    for video_name, file_map in sorted(video_groups.items()):
        consolidate(video_name, file_map)
    print(f"\n📁 Saved to: {OUTPUT_FOLDER}/")