#!/usr/bin/env python3
"""
Video Action Classification Script using Qwen2.5-VL
Analyzes 2-second video clips and classifies actions from a predefined set.
"""

import os
import torch
import json
import time
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict

# Predefined action classes
ACTIONS = ["walking", "sitting", "standing", "go upstair", "go downstair"]

def load_model(model_id):
    """Load the Qwen2.5-VL model"""
    print(f"🔄 Loading Qwen2.5-VL model: {model_id}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
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
    
    video_files.sort()  # Sort for consistent ordering
    print(f"📁 Found {len(video_files)} video files")
    return video_files

def get_model_device(model):
    """Get the main device of the model for input placement"""
    if hasattr(model, 'hf_device_map'):
        return list(model.hf_device_map.values())[0]
    else:
        return next(model.parameters()).device

def classify_video_action(model, processor, video_path: str) -> Dict:
    """Classify action in a single video clip"""
    
    # Create the prompt for action classification
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
    
    try:
        # Prepare inputs
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
        
        # Move inputs to the correct device
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=32,  # Short response for action classification
                do_sample=False,    # Deterministic for classification
                temperature=0.1
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clean up the response
        action = output_text[0].strip().lower()
        
        # Validate if the action is in our predefined list
        valid_action = None
        for predefined_action in ACTIONS:
            if predefined_action in action:
                valid_action = predefined_action
                break
        
        if valid_action is None:
            # If no exact match, try to find the closest match
            for predefined_action in ACTIONS:
                if any(word in action for word in predefined_action.split()):
                    valid_action = predefined_action
                    break
        
        return {
            "video_path": video_path,
            "raw_response": action,
            "classified_action": valid_action or "unknown",
            "confidence": "high" if valid_action else "low"
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Video Action Classification using Qwen2.5-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--video_folder", type=str, required=True,
                       help="Path to folder containing video files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSON file for results")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help="Model ID to use for classification")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.video_folder):
        parser.error(f"Video folder does not exist: {args.video_folder}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return args

def main():
    """Main function to analyze all video clips"""
    args = parse_args()
    
    print("🚀 Starting Video Action Classification")
    print("=" * 50)
    print(f"📁 Video folder: {args.video_folder}")
    print(f"💾 Output file: {args.output_file}")
    print(f"🤖 Model: {args.model_id}")
    print("=" * 50)
    
    # Load model
    model, processor = load_model(args.model_id)
    
    # Get video files
    video_files = get_video_files(args.video_folder)
    if not video_files:
        print("❌ No video files found!")
        return
    
    # Analyze each video
    results = []
    total_files = len(video_files)
    
    print(f"\n🎬 Analyzing {total_files} video clips...")
    print("=" * 50)
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n📹 Processing {i}/{total_files}: {video_name}")
        
        start_time = time.time()
        result = classify_video_action(model, processor, video_path)
        end_time = time.time()
        
        result["processing_time"] = end_time - start_time
        results.append(result)
        
        # Print result
        action = result["classified_action"]
        confidence = result["confidence"]
        print(f"   Action: {action} (confidence: {confidence})")
        print(f"   Time: {end_time - start_time:.2f}s")
        
        if "error" in result:
            print(f"   Error: {result['error']}")
    
    # Save results
    print(f"\n💾 Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n📊 Summary:")
    print("=" * 50)
    action_counts = {}
    for result in results:
        action = result["classified_action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in action_counts.items():
        print(f"   {action}: {count} clips")
    
    print(f"\n🎉 Analysis complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 