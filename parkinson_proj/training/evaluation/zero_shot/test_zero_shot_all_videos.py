#!/usr/bin/env python3
"""
Test zero-shot performance on all video files in the original folder
"""

import os
import json
import torch
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Predefined action classes
ACTIONS = ["walking", "sitting", "standing", "go upstair", "go downstair"]

def load_model(model_id):
    """Load the specified model"""
    print(f"🔄 Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("✅ Model loaded successfully!")
    return model, processor

def get_video_files(folder_path: str):
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
    
    video_files.sort()  # Sort for consistent ordering
    print(f"📁 Found {len(video_files)} video files")
    return video_files

def get_model_device(model):
    """Get the main device of the model for input placement"""
    if hasattr(model, 'hf_device_map'):
        return list(model.hf_device_map.values())[0]
    else:
        return next(model.parameters()).device

def extract_prediction(response_text):
    """Extract action prediction from model response."""
    response = response_text.strip().lower()
    
    # Try exact match first
    for action in ACTIONS:
        if action in response:
            return action
    
    # Try partial matches
    for action in ACTIONS:
        if any(word in response for word in action.split()):
            return action
    
    return "unknown"

def evaluate_single_video(model, processor, video_path):
    """Evaluate a single video file."""
    try:
        # Use the exact same prompt as the 78.32% results
        prompt = f"""Analyze this video clip and classify the action being performed. 
Choose ONLY ONE action from this list: {', '.join(ACTIONS)}

Respond with just the action name, nothing else."""

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
        
        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        # Move to device
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract prediction
        predicted_action = extract_prediction(response_text)
        
        return {
            "video_path": video_path,
            "predicted_action": predicted_action,
            "raw_response": response_text,
        }
        
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {e}")
        return {
            "video_path": video_path,
            "predicted_action": "error",
            "raw_response": "",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test zero-shot performance on all videos")
    parser.add_argument("--video_folder", type=str, default="/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/video_0",
                       help="Path to video folder")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model ID to test")
    parser.add_argument("--output_file", type=str, default="zero_shot_results_7B.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing Zero-Shot Performance: {args.model_id}")
    print("=" * 50)
    
    # Load model
    model, processor = load_model(args.model_id)
    
    # Get video files
    video_files = get_video_files(args.video_folder)
    if not video_files:
        print("❌ No video files found!")
        return
    
    # Evaluate all videos
    print(f"\n🎬 Evaluating {len(video_files)} videos...")
    results = []
    
    for video_path in tqdm(video_files, desc="Evaluating"):
        result = evaluate_single_video(model, processor, video_path)
        results.append(result)
    
    # Save results
    print(f"\n💾 Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n📊 Summary:")
    print("=" * 50)
    action_counts = defaultdict(int)
    for result in results:
        action = result["predicted_action"]
        action_counts[action] += 1
    
    total_videos = len(results)
    print(f"Total videos processed: {total_videos}")
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / total_videos) * 100
        print(f"  {action:12s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n🎉 Evaluation complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 