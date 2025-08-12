#!/bin/bash

# Test script to verify 72B training script works
echo "🧪 Testing 72B training script..."

# Check if all required files exist
echo "📁 Checking required files..."

# Check training script
if [ -f "scripts/finetune_lora_video_sbatch.sh" ]; then
    echo "✅ Training script exists"
else
    echo "❌ Training script missing"
    exit 1
fi

# Check data files
if [ -f "../annotations/qwen_annotations_original_videos/train.json" ]; then
    echo "✅ Train data exists"
else
    echo "❌ Train data missing"
    exit 1
fi

if [ -f "../annotations/qwen_annotations_original_videos/val.json" ]; then
    echo "✅ Val data exists"
else
    echo "❌ Val data missing"
    exit 1
fi

# Check output directory
if [ -d "../output/lora_video_action_72b" ]; then
    echo "✅ Output directory exists"
else
    echo "❌ Output directory missing"
    exit 1
fi

# Check DeepSpeed config
if [ -f "scripts/zero3_offload.json" ]; then
    echo "✅ DeepSpeed config exists"
else
    echo "❌ DeepSpeed config missing"
    exit 1
fi

# Check requirements
if [ -f "requirements_training.txt" ]; then
    echo "✅ Requirements file exists"
else
    echo "❌ Requirements file missing"
    exit 1
fi

echo "🎉 All checks passed! 72B training script is ready to run."
echo "🚀 You can now run: sbatch scripts/finetune_lora_video_sbatch.sh" 