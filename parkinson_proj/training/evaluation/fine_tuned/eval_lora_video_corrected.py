#!/usr/bin/env python3
"""
Corrected evaluation script for Qwen2.5-VL LoRA video action classification model.
Matches the exact conditions of the previous 78.32% accuracy results.
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Predefined action classes (matching the 78.32% results)
ACTIONS = ["walking", "sitting", "standing", "go upstair", "go downstair"]

def load_trained_model(model_path, base_model_id="Qwen/Qwen2.5-VL-7B-Instruct", use_base_model=False, use_merged_model=False):
    """Load the trained LoRA model, base model, or merged model."""
    print(f"🔄 Loading base model: {base_model_id}")
    
    if use_merged_model:
        print(f"🔄 Loading merged model from: {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif use_base_model:
        print("🔄 Using base model (no LoRA adapter)")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(base_model_id)
    else:
        print(f"🔄 Loading LoRA adapter from: {model_path}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        processor = AutoProcessor.from_pretrained(base_model_id)
    
    model.eval()
    return model, processor

def load_eval_data(test_json_path, val_json_path=None):
    """Load evaluation dataset (test + val)."""
    print(f"📊 Loading test data from: {test_json_path}")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"📊 Loaded {len(test_data)} test samples")
    
    eval_data = test_data
    
    if val_json_path:
        print(f"📊 Loading val data from: {val_json_path}")
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)
        print(f"📊 Loaded {len(val_data)} val samples")
        eval_data.extend(val_data)
    
    print(f"📊 Total evaluation samples: {len(eval_data)}")
    return eval_data

def extract_prediction(response_text):
    """Extract action prediction from model response (matching 78.32% approach)."""
    # Clean and normalize the response
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

def get_model_device(model):
    """Get the main device of the model for input placement"""
    if hasattr(model, 'hf_device_map'):
        return list(model.hf_device_map.values())[0]
    else:
        return next(model.parameters()).device

def evaluate_single_sample(model, processor, sample, image_folder):
    """Evaluate a single test sample (matching 78.32% approach)."""
    try:
        # Prepare the input
        video_path = os.path.join(image_folder, sample["video"])
        
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
        
        # Generate response (exact same settings as 78.32% approach)
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
        true_action = sample["conversations"][1]["value"].strip().lower()
        
        # Map action labels to match the 78.32% results
        action_mapping = {
            "upstair": "go upstair",
            "downstair": "go downstair"
        }
        true_action = action_mapping.get(true_action, true_action)
        
        return {
            "id": sample["id"],
            "video_path": sample["video"],
            "true_action": true_action,
            "predicted_action": predicted_action,
            "raw_response": response_text,
            "correct": predicted_action == true_action
        }
        
    except Exception as e:
        print(f"❌ Error processing sample {sample['id']}: {e}")
        return {
            "id": sample["id"],
            "video_path": sample["video"],
            "true_action": sample["conversations"][1]["value"].strip().lower(),
            "predicted_action": "error",
            "raw_response": "",
            "correct": False,
            "error": str(e)
        }

def calculate_metrics(results):
    """Calculate evaluation metrics."""
    # Overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    
    # Per-class metrics
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        true_action = result["true_action"]
        class_stats[true_action]["total"] += 1
        if result["correct"]:
            class_stats[true_action]["correct"] += 1
    
    per_class_accuracy = {}
    for action, stats in class_stats.items():
        per_class_accuracy[action] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # Confusion matrix
    y_true = [r["true_action"] for r in results]
    y_pred = [r["predicted_action"] for r in results]
    
    # Get unique classes
    classes = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    return {
        "overall_accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "classes": classes
    }

def convert_to_pd_insighter_format(results, fps=30):
    """Convert test predictions to PD insighter format.
    
    Args:
        results: List of prediction results from test_predictions.json
        fps: Frames per second for calculating frame numbers (default: 30)
        
    Returns:
        Dict: PD insighter format with action segments
    """
    # Sort results by clip number to ensure sequential order
    def extract_clip_number(clip_id):
        # Extract number from "video_0_clip_340" -> 340
        parts = clip_id.split('_')
        return int(parts[-1]) if parts[-1].isdigit() else 0
    
    sorted_results = sorted(results, key=lambda x: extract_clip_number(x["id"]))
    
    # Group consecutive clips with same predicted action
    pd_format = {}
    
    if not sorted_results:
        return pd_format
    
    current_action = sorted_results[0]["predicted_action"]
    current_start_clip = extract_clip_number(sorted_results[0]["id"])
    current_end_clip = current_start_clip
    
    for i in range(1, len(sorted_results)):
        clip_num = extract_clip_number(sorted_results[i]["id"])
        predicted_action = sorted_results[i]["predicted_action"]
        
        # Check if this continues the current action segment
        if predicted_action == current_action and clip_num == current_end_clip + 1:
            # Continue current segment
            current_end_clip = clip_num
        else:
            # End current segment and start new one
            # Calculate frame numbers (assuming each clip is 1 second at given fps)
            start_frame = current_start_clip * fps
            end_frame = (current_end_clip + 1) * fps - 1
            
            # Add to PD format
            if current_action not in pd_format:
                pd_format[current_action] = []
            
            pd_format[current_action].append({
                "Starting frame": start_frame,
                "Ending frame": end_frame
            })
            
            # Start new segment
            current_action = predicted_action
            current_start_clip = clip_num
            current_end_clip = clip_num
    
    # Don't forget the last segment
    start_frame = current_start_clip * fps
    end_frame = (current_end_clip + 1) * fps - 1
    
    if current_action not in pd_format:
        pd_format[current_action] = []
    
    pd_format[current_action].append({
        "Starting frame": start_frame,
        "Ending frame": end_frame
    })
    
    return pd_format

def save_results(results, metrics, output_dir):
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw predictions
    predictions_file = os.path.join(output_dir, "test_predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save PD insighter format predictions
    pd_format = convert_to_pd_insighter_format(results, fps=30)
    pd_file = os.path.join(output_dir, "pd_insighter_format_predictions.json")
    with open(pd_file, 'w') as f:
        json.dump(pd_format, f, indent=2)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "test_accuracy.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["classes"],
        columns=metrics["classes"]
    )
    cm_file = os.path.join(output_dir, "test_confusion_matrix.csv")
    cm_df.to_csv(cm_file)
    
    # Save per-class results
    per_class_data = []
    for action, stats in metrics["per_class_accuracy"].items():
        per_class_data.append({
            "action": action,
            "accuracy": stats,
            "total_samples": sum(1 for r in results if r["true_action"] == action),
            "correct_predictions": sum(1 for r in results if r["true_action"] == action and r["correct"])
        })
    
    per_class_df = pd.DataFrame(per_class_data)
    per_class_file = os.path.join(output_dir, "test_per_class_results.csv")
    per_class_df.to_csv(per_class_file, index=False)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 PD insighter format saved to: {pd_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA model on test set (corrected)")
    parser.add_argument("--model_path", type=str, default="../../../../output/lora_video_action",
                       help="Path to trained model directory")
    parser.add_argument("--test_data", type=str, default="../../../../annotations/qwen_annotations_original_videos/test.json",
                       help="Path to test dataset JSON")
    parser.add_argument("--val_data", type=str, default="../../../../annotations/qwen_annotations_original_videos/val.json",
                       help="Path to validation dataset JSON (optional)")
    parser.add_argument("--image_folder", type=str, default="../../../../data",
                       help="Path to image/video folder")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (if not specified, uses model_path/eval_results)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Base model ID")
    parser.add_argument("--use_base_model", action="store_true",
                       help="Use base model instead of trained LoRA model")
    parser.add_argument("--use_merged_model", action="store_true",
                       help="Use merged model (LoRA weights merged with base model)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Corrected LoRA Model Evaluation")
    print("=" * 50)
    print("📝 Using exact same conditions as 78.32% accuracy results")
    
    # Load model
    model, processor = load_trained_model(args.model_path, args.base_model, args.use_base_model, args.use_merged_model)
    
    # Load evaluation data (test + val)
    eval_data = load_eval_data(args.test_data, args.val_data)
    
    # Evaluate all samples
    print(f"\n🎬 Evaluating {len(eval_data)} samples (test + val)...")
    results = []
    
    for sample in tqdm(eval_data, desc="Evaluating"):
        result = evaluate_single_sample(model, processor, sample, args.image_folder)
        results.append(result)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Print summary
    print("\n📈 Evaluation Results:")
    print("=" * 50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
    print(f"\nPer-class Accuracy:")
    for action, acc in metrics['per_class_accuracy'].items():
        print(f"  {action:12s}: {acc:.4f}")
    
    # Set output directory based on model path and type
    if args.output_dir:
        # Use user-specified output directory
        output_dir = args.output_dir
        print(f"📝 Using custom output directory: {output_dir}")
    else:
        # Use model path to determine output directory
        if args.use_base_model:
            output_dir = os.path.join(args.model_path, "eval_results_base_model")
            print(f"📝 Using base model - results will be saved to: {output_dir}")
        elif args.use_merged_model:
            output_dir = os.path.join(args.model_path, "eval_results")
            print(f"📝 Using merged model - results will be saved to: {output_dir}")
        else:
            output_dir = os.path.join(args.model_path, "eval_results_lora")
            print(f"📝 Using LoRA model - results will be saved to: {output_dir}")
    
    # Save results
    save_results(results, metrics, output_dir)
    
    print(f"\n🎉 Evaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 