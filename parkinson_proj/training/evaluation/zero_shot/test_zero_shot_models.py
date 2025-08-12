#!/usr/bin/env python3
"""
Test zero-shot performance of both 7B and 72B models
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

def evaluate_single_sample(model, processor, sample, image_folder):
    """Evaluate a single test sample."""
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
        true_action = sample["conversations"][1]["value"].strip().lower()
        
        # Map action labels
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
    
    return {
        "overall_accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Test zero-shot performance of different models")
    parser.add_argument("--test_data", type=str, default="annotations/qwen_annotations/test.json",
                       help="Path to test dataset JSON")
    parser.add_argument("--val_data", type=str, default="annotations/qwen_annotations/val.json",
                       help="Path to validation dataset JSON")
    parser.add_argument("--image_folder", type=str, default="data",
                       help="Path to image/video folder")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model ID to test")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing Zero-Shot Performance: {args.model_id}")
    print("=" * 50)
    
    # Load model
    model, processor = load_model(args.model_id)
    
    # Load evaluation data
    print(f"📊 Loading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"📊 Loaded {len(test_data)} test samples")
    
    print(f"📊 Loading val data from: {args.val_data}")
    with open(args.val_data, 'r') as f:
        val_data = json.load(f)
    print(f"📊 Loaded {len(val_data)} val samples")
    
    eval_data = test_data + val_data
    print(f"📊 Total evaluation samples: {len(eval_data)}")
    
    # Evaluate all samples
    print(f"\n🎬 Evaluating {len(eval_data)} samples...")
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
    print(f"Model: {args.model_id}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
    print(f"\nPer-class Accuracy:")
    for action, acc in metrics['per_class_accuracy'].items():
        print(f"  {action:12s}: {acc:.4f}")
    
    print(f"\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main() 