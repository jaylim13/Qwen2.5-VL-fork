import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Helper to get all video files in data/video_*/
def get_video_files(data_root):
    video_files = []
    for video_dir in sorted(Path(data_root).glob("video_*/")):
        for video_file in video_dir.glob("clip_*.mp4"):
            video_files.append(video_file)
    return video_files

def get_video_stats(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count

def modify_video(input_path, output_path, target_width, target_height, target_fps):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {input_path}")
        return False
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_width, target_height))
    frame_idx = 0
    frame_interval = int(round(orig_fps / target_fps)) if target_fps < orig_fps else 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_fps < orig_fps and frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_resized = cv2.resize(frame, (target_width, target_height))
        out.write(frame_resized)
        frame_idx += 1
    cap.release()
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Check and optionally modify video data stats.")
    parser.add_argument('--data_root', type=str, default='data', help='Root folder containing video_*/')
    parser.add_argument('--modify', action='store_true', help='If set, modify videos to target params')
    parser.add_argument('--target_width', type=int, default=None, help='Target width for resizing')
    parser.add_argument('--target_height', type=int, default=None, help='Target height for resizing')
    parser.add_argument('--target_fps', type=float, default=None, help='Target FPS for resampling')
    parser.add_argument('--output_dir', type=str, default=None, help='If set, save modified videos here (else defaults to data/qwen_data/)')
    args = parser.parse_args()

    video_files = get_video_files(args.data_root)
    print(f"Found {len(video_files)} video files.")

    widths, heights, fpss, frame_counts = [], [], [], []
    for video_file in tqdm(video_files, desc="Analyzing videos"):
        stats = get_video_stats(video_file)
        if stats is None:
            print(f"[WARN] Could not read {video_file}")
            continue
        w, h, f, n = stats
        widths.append(w)
        heights.append(h)
        fpss.append(f)
        frame_counts.append(n)

    if widths:
        print(f"Width: min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
        print(f"Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
        print(f"FPS: min={min(fpss):.2f}, max={max(fpss):.2f}, mean={sum(fpss)/len(fpss):.2f}")
        print(f"Frame count: min={min(frame_counts)}, max={max(frame_counts)}, mean={sum(frame_counts)/len(frame_counts):.1f}")
    else:
        print("No readable videos found.")

    if args.modify:
        assert args.target_width and args.target_height and args.target_fps, "Must specify --target_width, --target_height, --target_fps when --modify is set."
        out_dir = Path(args.output_dir) if args.output_dir else Path(args.data_root) / "qwen_data"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Modified videos will be saved to: {out_dir}")
        for video_file in tqdm(video_files, desc="Modifying videos"):
            rel_path = video_file.relative_to(args.data_root)
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            success = modify_video(video_file, out_path, args.target_width, args.target_height, args.target_fps)
            if not success:
                print(f"[WARN] Failed to modify {video_file}")
        print("Modification complete.")

if __name__ == "__main__":
    main() 