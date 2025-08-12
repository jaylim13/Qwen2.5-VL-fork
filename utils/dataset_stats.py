#!/usr/bin/env python3
"""
Dataset Statistics Script for Qwen2.5-VL Video Action Classification
Analyzes train/val/test datasets and generates comprehensive statistics reports.
"""

import os
import json
import argparse
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(json_path):
    """Load dataset from JSON file."""
    print(f"📊 Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"📊 Loaded {len(data)} samples")
    return data

def extract_video_source(video_path):
    """Extract video source from path (e.g., video_0, video_1, etc.)."""
    # Extract the video folder name from path like "qwen_data/video_0/clip_123.mp4"
    parts = video_path.split('/')
    for part in parts:
        if part.startswith('video_'):
            return part
    return "unknown"

def analyze_dataset(data, split_name):
    """Analyze a single dataset split."""
    print(f"\n🔍 Analyzing {split_name} dataset...")
    
    # Basic statistics
    total_samples = len(data)
    
    # Label distribution
    label_counts = Counter()
    video_source_counts = Counter()
    
    for sample in data:
        # Extract label
        label = sample["conversations"][1]["value"].strip().lower()
        label_counts[label] += 1
        
        # Extract video source
        video_source = extract_video_source(sample["video"])
        video_source_counts[video_source] += 1
    
    # Calculate percentages
    label_percentages = {}
    for label, count in label_counts.items():
        label_percentages[label] = (count / total_samples) * 100
    
    video_source_percentages = {}
    for source, count in video_source_counts.items():
        video_source_percentages[source] = (count / total_samples) * 100
    
    return {
        "split_name": split_name,
        "total_samples": total_samples,
        "label_counts": dict(label_counts),
        "label_percentages": label_percentages,
        "video_source_counts": dict(video_source_counts),
        "video_source_percentages": video_source_percentages,
        "unique_labels": len(label_counts),
        "unique_video_sources": len(video_source_counts)
    }

def create_comparison_dataframe(stats_list):
    """Create a comparison dataframe across all splits."""
    comparison_data = []
    
    for stats in stats_list:
        for label, count in stats["label_counts"].items():
            comparison_data.append({
                "split": stats["split_name"],
                "label": label,
                "count": count,
                "percentage": stats["label_percentages"][label]
            })
    
    return pd.DataFrame(comparison_data)

def save_statistics(stats_list, output_dir):
    """Save all statistics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall dataset summary
    summary_data = []
    for stats in stats_list:
        summary_data.append({
            "split": stats["split_name"],
            "total_samples": stats["total_samples"],
            "unique_labels": stats["unique_labels"],
            "unique_video_sources": stats["unique_video_sources"],
            "most_common_label": max(stats["label_counts"], key=stats["label_counts"].get),
            "most_common_video_source": max(stats["video_source_counts"], key=stats["video_source_counts"].get)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Dataset summary saved to: {summary_file}")
    
    # 2. Label distribution
    label_data = []
    for stats in stats_list:
        for label, count in stats["label_counts"].items():
            label_data.append({
                "split": stats["split_name"],
                "label": label,
                "count": count,
                "percentage": stats["label_percentages"][label]
            })
    
    label_df = pd.DataFrame(label_data)
    label_file = os.path.join(output_dir, "label_distribution.csv")
    label_df.to_csv(label_file, index=False)
    print(f"💾 Label distribution saved to: {label_file}")
    
    # 3. Video source distribution
    source_data = []
    for stats in stats_list:
        for source, count in stats["video_source_counts"].items():
            source_data.append({
                "split": stats["split_name"],
                "video_source": source,
                "count": count,
                "percentage": stats["video_source_percentages"][source]
            })
    
    source_df = pd.DataFrame(source_data)
    source_file = os.path.join(output_dir, "video_source_distribution.csv")
    source_df.to_csv(source_file, index=False)
    print(f"💾 Video source distribution saved to: {source_file}")
    
    # 4. Detailed statistics JSON
    detailed_stats = {
        "dataset_overview": {
            "total_samples": sum(stats["total_samples"] for stats in stats_list),
            "splits": [stats["split_name"] for stats in stats_list],
            "all_labels": list(set().union(*[set(stats["label_counts"].keys()) for stats in stats_list])),
            "all_video_sources": list(set().union(*[set(stats["video_source_counts"].keys()) for stats in stats_list]))
        },
        "split_details": {stats["split_name"]: stats for stats in stats_list}
    }
    
    detailed_file = os.path.join(output_dir, "detailed_statistics.json")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    print(f"💾 Detailed statistics saved to: {detailed_file}")
    
    # 5. Balance analysis report
    balance_report = generate_balance_report(stats_list)
    balance_file = os.path.join(output_dir, "dataset_balance_report.txt")
    with open(balance_file, 'w') as f:
        f.write(balance_report)
    print(f"💾 Balance report saved to: {balance_file}")
    
    return {
        "summary_file": summary_file,
        "label_file": label_file,
        "source_file": source_file,
        "detailed_file": detailed_file,
        "balance_file": balance_file
    }

def generate_balance_report(stats_list):
    """Generate a balance analysis report."""
    report = "DATASET BALANCE ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"
    
    # Overall balance
    total_samples = sum(stats["total_samples"] for stats in stats_list)
    report += f"TOTAL SAMPLES: {total_samples:,}\n"
    report += f"SPLITS: {', '.join(stats['split_name'] for stats in stats_list)}\n\n"
    
    # Label balance analysis
    all_labels = set().union(*[set(stats["label_counts"].keys()) for stats in stats_list])
    report += "LABEL DISTRIBUTION ANALYSIS:\n"
    report += "-" * 30 + "\n"
    
    for label in sorted(all_labels):
        report += f"\n{label.upper()}:\n"
        for stats in stats_list:
            count = stats["label_counts"].get(label, 0)
            percentage = stats["label_percentages"].get(label, 0)
            report += f"  {stats['split_name']}: {count:,} samples ({percentage:.1f}%)\n"
    
    # Video source balance
    all_sources = set().union(*[set(stats["video_source_counts"].keys()) for stats in stats_list])
    report += f"\nVIDEO SOURCE DISTRIBUTION:\n"
    report += "-" * 30 + "\n"
    
    for source in sorted(all_sources):
        report += f"\n{source.upper()}:\n"
        for stats in stats_list:
            count = stats["video_source_counts"].get(source, 0)
            percentage = stats["video_source_percentages"].get(source, 0)
            report += f"  {stats['split_name']}: {count:,} samples ({percentage:.1f}%)\n"
    
    # Balance assessment
    report += f"\nBALANCE ASSESSMENT:\n"
    report += "-" * 20 + "\n"
    
    # Check for severe imbalances
    for stats in stats_list:
        label_counts = list(stats["label_counts"].values())
        if label_counts:
            min_count = min(label_counts)
            max_count = max(label_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 5:
                report += f"⚠️  {stats['split_name']}: Severe label imbalance (ratio: {imbalance_ratio:.1f})\n"
            elif imbalance_ratio > 2:
                report += f"⚠️  {stats['split_name']}: Moderate label imbalance (ratio: {imbalance_ratio:.1f})\n"
            else:
                report += f"✅ {stats['split_name']}: Good label balance (ratio: {imbalance_ratio:.1f})\n"
    
    return report

def print_summary(stats_list):
    """Print a summary of the analysis."""
    print("\n📊 DATASET STATISTICS SUMMARY")
    print("=" * 50)
    
    for stats in stats_list:
        print(f"\n{stats['split_name'].upper()} DATASET:")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Unique labels: {stats['unique_labels']}")
        print(f"  Unique video sources: {stats['unique_video_sources']}")
        
        print(f"  Label distribution:")
        for label, count in sorted(stats["label_counts"].items()):
            percentage = stats["label_percentages"][label]
            print(f"    {label:12s}: {count:6,} ({percentage:5.1f}%)")
        
        print(f"  Video source distribution:")
        for source, count in sorted(stats["video_source_counts"].items()):
            percentage = stats["video_source_percentages"][source]
            print(f"    {source:12s}: {count:6,} ({percentage:5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics for video action classification")
    parser.add_argument("--train_data", type=str, default="../annotations/qwen_annotations/train.json",
                       help="Path to training dataset JSON")
    parser.add_argument("--val_data", type=str, default="../annotations/qwen_annotations/val.json",
                       help="Path to validation dataset JSON")
    parser.add_argument("--test_data", type=str, default="../annotations/qwen_annotations/test.json",
                       help="Path to test dataset JSON")
    parser.add_argument("--output_dir", type=str, default="../output/lora_video_action/dataset_stats",
                       help="Output directory for statistics")
    
    args = parser.parse_args()
    
    print("🚀 Starting Dataset Statistics Analysis")
    print("=" * 50)
    
    # Load all datasets
    datasets = []
    for split_name, data_path in [("train", args.train_data), ("val", args.val_data), ("test", args.test_data)]:
        if os.path.exists(data_path):
            data = load_dataset(data_path)
            stats = analyze_dataset(data, split_name)
            datasets.append((split_name, data, stats))
        else:
            print(f"⚠️  Warning: {data_path} not found, skipping {split_name} dataset")
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Extract statistics
    stats_list = [stats for _, _, stats in datasets]
    
    # Print summary
    print_summary(stats_list)
    
    # Save statistics
    print(f"\n💾 Saving statistics to: {args.output_dir}")
    saved_files = save_statistics(stats_list, args.output_dir)
    
    print(f"\n🎉 Analysis complete! Files saved:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")

if __name__ == "__main__":
    main() 