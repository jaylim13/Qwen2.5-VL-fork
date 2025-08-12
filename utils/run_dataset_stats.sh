#!/bin/bash

# ====== Dataset Statistics Analysis ======

# Data paths
TRAIN_DATA="./annotations/qwen_annotations/train.json"
VAL_DATA="./annotations/qwen_annotations/val.json"
TEST_DATA="./annotations/qwen_annotations/test.json"
OUTPUT_DIR="./output/lora_video_action/dataset_stats"

# Run dataset statistics
echo "🚀 Starting dataset statistics analysis..."
python utils/dataset_stats.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir $OUTPUT_DIR

echo "✅ Dataset statistics analysis complete!" 