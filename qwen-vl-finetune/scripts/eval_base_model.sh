#!/bin/bash

# ====== Qwen2.5-VL Base Model Evaluation ======

# Model and data paths
MODEL_PATH="../output/lora_video_action"
TEST_DATA="../annotations/qwen_annotations/test.json"
IMAGE_FOLDER="../data"
OUTPUT_DIR="../output/lora_video_action/eval_results"
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

# Run evaluation with base model
echo "🚀 Starting evaluation with base model..."
python scripts/eval_lora_video.py \
    --model_path $MODEL_PATH \
    --test_data $TEST_DATA \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL \
    --use_base_model

echo "✅ Base model evaluation complete!" 