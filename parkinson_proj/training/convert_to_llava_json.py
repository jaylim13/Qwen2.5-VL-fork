import os
import csv
import json
from pathlib import Path
import argparse

# Action label set (edit if needed)
ACTIONS = {"walking", "sitting", "standing", "upstair", "downstair", "go upstair", "go downstair"}

# Helper to check video file existence
def check_video_path(video_root, source_video, clip_id):
    # CSV: video_1_clip_504 -> data/video_1/clip_504.mp4
    # Map source_video (e.g., "video_1") to the actual folder name
    video_folder = source_video  # Use the source_video directly
    clip_num = clip_id.split("_")[-1]
    video_path = video_root / video_folder / f"clip_{clip_num}.mp4"
    return video_path

def convert_csv_to_llava_json(split, csv_dir, video_root, output_dir):
    csv_path = csv_dir / f"{split}.csv"
    output_path = output_dir / f"{split}.json"
    assert csv_path.exists(), f"CSV file not found: {csv_path}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entries = []
    for row in csv.DictReader(open(csv_path, newline='')):
        # Required keys
        for key in ["video_id", "action_label", "source_video"]:
            assert key in row, f"Missing key {key} in CSV row: {row}"
        video_id = row["video_id"]
        action_label = row["action_label"].strip().lower()
        source_video = row["source_video"]
        # Validate action label
        assert any(action_label in a or a in action_label for a in ACTIONS), f"Unknown action label: {action_label}"
        # Map to video path
        video_path = check_video_path(video_root, source_video, video_id)
        assert video_path.exists(), f"Video file does not exist: {video_path} (from {video_id})"
        # Compose LLaVA-style entry
        entry = {
            "id": video_id,
            "video": str(video_path.relative_to(video_root)),  # Remove data/ prefix
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nWhat action is being performed in this video?"
                },
                {
                    "from": "gpt",
                    "value": action_label
                }
            ]
        }
        entries.append(entry)
    # Write JSON
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✅ Converted {csv_path} to {output_path} ({len(entries)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Convert mode2 CSV to LLaVA-style JSON for Qwen2-VL video finetuning.")
    parser.add_argument('--csv_dir', type=str, default='annotations/mode2', help='Directory containing train/val/test.csv')
    parser.add_argument('--video_root', type=str, default='data', help='Root directory for video files (e.g., data or data/qwen_data)')
    parser.add_argument('--output_dir', type=str, default='annotations/qwen_annotations_original_videos', help='Where to save the output JSON files')
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    video_root = Path(args.video_root)
    output_dir = Path(args.output_dir)
    print(f"[INFO] Using video root: {video_root}")

    for split in ["train", "val", "test"]:
        convert_csv_to_llava_json(split, csv_dir, video_root, output_dir)

if __name__ == "__main__":
    main() 