import csv
import shutil
from pathlib import Path

# -------- CONFIG -------- #
VIDEO_DIRS = [
    "data/video_h",
    "data/video_j",
    "data/video_j3",
    "data/video_j5",
]

ANNOTATION_DIRS = [
    "annotations/video_h",
    "annotations/video_j",
    "annotations/video_j3",
    "annotations/video_j5",
]

OUTPUT_DIR = "data/flat_dataset"
VIDEO_OUT = Path(OUTPUT_DIR) / "videos"
LABEL_OUT = Path(OUTPUT_DIR) / "labels.csv"
# ------------------------ #

def main():
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    clip_idx = 0

    for video_dir, ann_dir in zip(VIDEO_DIRS, ANNOTATION_DIRS):
        video_dir = Path(video_dir)
        ann_dir = Path(ann_dir)

        label_file = ann_dir / "labels.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"Missing labels.csv in {ann_dir}")

        with open(label_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                old_name = row["video_id"]
                action = row["action_label"]

                # Strip any prefix up to last underscore
                if "_" in old_name:
                    old_name = old_name.split("_")[-1]

                # Ensure .mp4 extension
                if not old_name.endswith(".mp4"):
                    old_name += ".mp4"

                old_path = video_dir / old_name
                if not old_path.exists():
                    # Try using the raw clip name from CSV as fallback
                    fallback_path = video_dir / (row["video_id"] + ".mp4")
                    if fallback_path.exists():
                        old_path = fallback_path
                    else:
                        raise FileNotFoundError(f"Missing clip: {old_path}")

                new_name = f"clip_{clip_idx:05d}.mp4"
                shutil.copy2(old_path, VIDEO_OUT / new_name)

                all_rows.append({
                    "video_id": new_name,
                    "action_label": action
                })

                clip_idx += 1

    # Write merged labels.csv
    with open(LABEL_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "action_label"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("✅ Dataset constructed successfully")
    print(f"   Total clips: {clip_idx}")
    print(f"   Videos directory: {VIDEO_OUT}")
    print(f"   Labels file: {LABEL_OUT}")

if __name__ == "__main__":
    main()
