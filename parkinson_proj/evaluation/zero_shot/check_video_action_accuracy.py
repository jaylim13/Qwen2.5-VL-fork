#!/usr/bin/env python3
"""
Check classification accuracy for Qwen2.5-VL video action results.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# Argument parsing
parser = argparse.ArgumentParser(description="Check video action classification accuracy.")
parser.add_argument('--split', type=str, default='all', choices=['all', 'test'], help="Which set to evaluate: all (default) or test")
parser.add_argument('--results_json', type=str, required=True, help="Path to results JSON file")
args = parser.parse_args()

# Paths
RESULTS_JSON = args.results_json
ANNOT_DIR = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/annotations/mode2"
ANNOT_FILES = ["train.csv", "val.csv", "test.csv"]

# 1. Load all ground truth annotations
gt_labels = {}
test_set_ids = set()
for annot_file in ANNOT_FILES:
    annot_path = os.path.join(ANNOT_DIR, annot_file)
    with open(annot_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            label = row["action_label"].strip().lower()
            gt_labels[video_id] = label
            if annot_file == "test.csv":
                test_set_ids.add(video_id)

print(f"Loaded {len(gt_labels)} ground truth labels.")

# 2. Load model predictions
with open(RESULTS_JSON, "r") as f:
    preds = json.load(f)

# Print a few example IDs for debugging
def extract_pred_video_id(pred):
    video_path = Path(pred["video_path"])
    if video_path.parent.name.startswith("video_"):
        return f"{video_path.parent.name}_{video_path.stem}"
    else:
        return video_path.stem

print("\nExample ground truth video_ids:", list(gt_labels.keys())[:5])
print("Example predicted video_ids:", [extract_pred_video_id(pred) for pred in preds[:5]])

# 3. Match predictions to ground truth
if args.split == 'test':
    print("\nEvaluating only on test set.")
else:
    print("\nEvaluating on all data (train+val+test).")

total = 0
correct = 0
per_class = defaultdict(lambda: {"correct": 0, "total": 0})

for pred in preds:
    # Extract video_id as subfolder_clipname (without extension)
    video_id = extract_pred_video_id(pred)
    if args.split == 'test' and video_id not in test_set_ids:
        continue
    # Handle both field names from different result formats
    if "classified_action" in pred:
        pred_label = pred["classified_action"].strip().lower()
    elif "predicted_action" in pred:
        pred_label = pred["predicted_action"].strip().lower()
    else:
        print(f"Warning: No action field found in prediction: {pred}")
        continue
    gt_label = gt_labels.get(video_id, None)
    if gt_label is None:
        continue  # skip if no ground truth

    total += 1
    per_class[gt_label]["total"] += 1
    # Allow for 'upstair' <-> 'go upstair' and 'downstair' <-> 'go downstair'
    if pred_label == gt_label or (gt_label == "upstair" and pred_label == "go upstair") or (gt_label == "downstair" and pred_label == "go downstair"):
        correct += 1
        per_class[gt_label]["correct"] += 1

# 4. Print results
if total == 0:
    print("\n❌ No predictions matched any ground truth video_id!")
    print("Check if the video_id extraction matches between predictions and ground truth.")
else:
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.2%}")

    print("\nPer-class accuracy:")
    for cls, stats in per_class.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cls:12s}: {stats['correct']}/{stats['total']} = {acc:.2%}") 