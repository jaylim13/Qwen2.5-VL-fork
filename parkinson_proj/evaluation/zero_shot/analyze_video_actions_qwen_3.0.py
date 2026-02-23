#!/usr/bin/env python3
"""
Video Action Classification Script using Qwen-3.0-VL
Analyzes 2-second video clips and classifies actions from a predefined set.
"""

import os
import torch
import json
import time
import argparse
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from typing import List, Dict

# Predefined action classes
ACTIONS = ["walking", "sitting", "standing", "go upstair", "go downstair"]
SLAM_SUMMARY_FILE = "data/video_h/henry_slam_clip_summaries.json"
# GUIDELINES = """
# You must classify the video clip with corresponding textual data into exactly one of the following actions: ["walking", "sitting", "standing", "go upstair", "go downstair"]

# "walking": noticeable horizontal displacement and speed, litle-to-no vertical displacement
# "sitting": little-to-no horizontal displacement, negative vertical displacement may have ocurred
# "standing": little-to-no horizontal displacement, positive vertical displacement may have occured, the user is NOT progressing forward
# "go upstair": noticeable horizontal displacement, positive vertical displacement, and noticeable speed, there are stairs or steps in view, user is ascending
# "go downstair": noticeable horizontal displacement, negative vertical displacement, and noticeable speed, there are stairs or steps in view,user is descending

# """

def load_slam_summaries(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {d["clip_index"]: d["slam_text_summary"] for d in data}

def load_model(model_id):
    """Load the Qwen-3.0-VL model"""
    print(f"🔄 Loading Qwen-3.0-VL model: {model_id}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("✅ Model loaded successfully!")
    return model, processor

def get_video_files(folder_path: str) -> List[str]:
    """Get all video files from the folder"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []

    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder {folder_path} does not exist!")
        return video_files

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(str(file))

    video_files.sort()
    print(f"📁 Found {len(video_files)} video files")
    return video_files

def get_model_device(model):
    """Get the main device of the model for input placement"""
    if hasattr(model, 'hf_device_map'):
        return list(model.hf_device_map.values())[0]
    return next(model.parameters()).device

def classify_video_action(model, processor, video_path: str, slam_text: str) -> Dict:
    """Classify action in a single video clip"""

    prompt = (
        "You are classifying a first-person video clip using BOTH the video and the motion data below.\n\n"
        "Motion data summary:\n"
        f"{slam_text}\n\n"
        "Classify the action into EXACTLY ONE of the following classes:\n"
        f"{', '.join(ACTIONS)}\n\n"
        "Action definitions:\n"
        "- walking: clear horizontal translation over time, moderate speed, minimal vertical displacement\n"
        "- sitting: very low speed, minimal horizontal translation, negative vertical displacement may occur\n"
        "- standing: little to no horizontal translation, may include rotation in place (looking around), "
        "low linear displacement, user is NOT progressing forward\n"
        "- go upstair: forward translation with noticeable positive vertical displacement and sustained speed, stairs visible\n"
        "- go downstair: forward translation with noticeable negative vertical displacement and sustained speed, stairs visible\n\n"
        "Important rules:\n"
        "- Rotation in place without forward translation indicates standing, NOT walking.\n"
        "- Use motion data to determine displacement and speed.\n"
        "- Respond with ONLY the action name, nothing else."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            if len(video_kwargs["fps"]) == 1:
                video_kwargs["fps"] = video_kwargs["fps"][0]
            elif len(video_kwargs["fps"]) == 0:
                video_kwargs["fps"] = None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )

        inputs = inputs.to(get_model_device(model))

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,   # shorter = cleaner classification
                do_sample=False,
                temperature=0.0
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        raw_action = output_text[0].strip().lower()
        raw_action = raw_action.replace(".", "").replace(",", "")

        # STRICT label matching (recommended for eval)
        classified_action = raw_action if raw_action in ACTIONS else "unknown"

        return {
            "video_path": video_path,
            "raw_response": raw_action,
            "classified_action": classified_action,
            "confidence": "high" if classified_action != "unknown" else "low"
        }

    except Exception as e:
        return {
            "video_path": video_path,
            "raw_response": "",
            "classified_action": "error",
            "confidence": "error",
            "error": str(e)
        }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Action Classification using Qwen-3.0-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID to use for classification"
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        parser.error(f"Video folder does not exist: {args.video_folder}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    return args

def main():
    if not os.path.exists(SLAM_SUMMARY_FILE):
        raise FileNotFoundError(f"SLAM summary file not found: {SLAM_SUMMARY_FILE}")
    slam_summaries = load_slam_summaries(SLAM_SUMMARY_FILE)
    args = parse_args()
    print("🚀 Starting Video Action Classification (Qwen-3.0-VL)")
    print("=" * 50)
    print(f"📁 Video folder: {args.video_folder}")
    print(f"💾 Output file: {args.output_file}")
    print(f"🤖 Model: {args.model_id}")
    print("=" * 50)

    model, processor = load_model(args.model_id)
    video_files = get_video_files(args.video_folder)

    if not video_files:
        print("❌ No video files found!")
        return

    results = []

    for i, video_path in enumerate(video_files, 1):
        clip_index = f"clip_{i:04d}"
        slam_text = slam_summaries.get(clip_index, "No motion data available.")

        print(f"\n📹 Processing {i}/{len(video_files)}: {os.path.basename(video_path)}")
        start_time = time.time()
        result = classify_video_action(
            model,
            processor,
            video_path,
            slam_text
        )
        result["processing_time"] = time.time() - start_time
        results.append(result)

        print(f"   Action: {result['classified_action']}")
        print(f"   Time: {result['processing_time']:.2f}s")

    print(f"\n💾 Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
