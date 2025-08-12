#!/usr/bin/env python3
"""
Concatenate video_0 clips, overlay true/predicted labels, and output annotated video and error segments JSON.
"""

import os
import cv2
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create annotated videos with GT vs predicted labels")
    parser.add_argument("--pred_json", type=str, required=True,
                       help="Path to predictions JSON file")
    parser.add_argument("--gt_dir", type=str, default="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/annotations/mode2",
                       help="Path to ground truth annotations directory")
    parser.add_argument("--video_dir", type=str, default="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/video_0",
                       help="Path to video clips directory")
    parser.add_argument("--output_video", type=str, default="annotated_videos/video_0_annotated.mp4",
                       help="Output video file path")
    parser.add_argument("--output_segments", type=str, default="annotated_videos/video_0_segments.json",
                       help="Output segments JSON file path")
    return parser.parse_args()

# --- CONFIG ---
BAR_HEIGHT = 60  # pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
THICKNESS = 2

def main():
    args = parse_args()
    
    # --- Ensure output directory exists ---
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    # --- 1. Load ground truth labels for video_0 clips ---
    gt_labels = {}
    annot_files = ["train.csv", "val.csv", "test.csv"]
    for annot_file in annot_files:
        annot_path = os.path.join(args.gt_dir, annot_file)
        if os.path.exists(annot_path):
            with open(annot_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row["video_id"]
                    if vid.startswith("video_0_"):
                        gt_labels[vid] = row["action_label"].strip().lower()
    
    print(f"Loaded {len(gt_labels)} ground truth labels")
    
    # --- 2. Load predictions for video_0 clips ---
    with open(args.pred_json) as f:
        preds = json.load(f)
    
    # Map: video_id -> pred_label
    pred_labels = {}
    for pred in preds:
        # Handle different prediction formats
        if "video_path" in pred:
            video_path = Path(pred["video_path"])
            video_id = f"{video_path.parent.name}_{video_path.stem}"
            pred_key = "classified_action" if "classified_action" in pred else "predicted_action"
            pred_labels[video_id] = pred[pred_key].strip().lower()
        elif "video" in pred:
            # Handle format from eval_lora_video_corrected.py
            video_path = pred["video"]
            video_id = f"video_0_{Path(video_path).stem}"
            pred_labels[video_id] = pred["predicted_action"].strip().lower()
    
    print(f"Loaded {len(pred_labels)} predictions")
    
    # --- 3. Get sorted list of all video_0 clips ---
    clips = sorted([str(p) for p in Path(args.video_dir).glob("clip_*.mp4")])
    print(f"Found {len(clips)} video clips")
    
    # --- 4. Prepare output video writer ---
    # Get video properties from first clip
    cap0 = cv2.VideoCapture(clips[0])
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    
    out_height = height + BAR_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, out_height))
    
    # --- 5. Process and annotate all clips ---
    segments = []  # For JSON output
    frame_idx = 0
    print(f"\nProcessing {len(clips)} clips with progress bar...")
    for clip_path in tqdm(clips, desc="Clips", unit="clip"):
        clip_name = Path(clip_path).stem
        video_id = f"video_0_{clip_name}"
        true_label = gt_labels.get(video_id, "unknown")
        pred_label = pred_labels.get(video_id, "unknown")
        is_correct = (true_label == pred_label or
                      (true_label == "upstair" and pred_label == "go upstair") or
                      (true_label == "downstair" and pred_label == "go downstair"))
    
        # Open video
        cap = cv2.VideoCapture(clip_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = frame_idx
        for _ in tqdm(range(n_frames), desc=f"Frames ({clip_name})", leave=False, unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            # Add top bar
            bar = 255 * np.ones((BAR_HEIGHT, width, 3), dtype=np.uint8)
            # If wrong, make bar background light red and use black text for contrast
            if not is_correct:
                bar[:] = (180, 180, 255)
                gt_color = (0, 0, 0)  # black
                pred_color = (0, 0, 0)  # black
            else:
                gt_color = (0, 200, 0)  # green
                pred_color = (0, 140, 255)  # orange
            # True label
            cv2.putText(bar, f"GT: {true_label}", (20, BAR_HEIGHT - 20), FONT, FONT_SCALE, gt_color, THICKNESS, cv2.LINE_AA)
            # Pred label
            cv2.putText(bar, f"Pred: {pred_label}", (width//2, BAR_HEIGHT - 20), FONT, FONT_SCALE, pred_color, THICKNESS, cv2.LINE_AA)
            # Stack bar on top
            annotated = np.vstack([bar, frame])
            writer.write(annotated)
            frame_idx += 1
        cap.release()
        # Record segment info
        segments.append({
            "video_id": video_id,
            "clip_path": clip_path,
            "start_frame": start_frame,
            "end_frame": frame_idx-1,
            "true_label": true_label,
            "pred_label": pred_label,
            "is_correct": is_correct
        })
    
    writer.release()
    
    # --- 6. Write segments JSON ---
    with open(args.output_segments, "w") as f:
        json.dump(segments, f, indent=2)
    
    print(f"\n✅ Annotated video saved to: {args.output_video}")
    print(f"✅ Segment info saved to: {args.output_segments}")

if __name__ == "__main__":
    main() 