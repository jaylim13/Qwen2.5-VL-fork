#!/bin/bash
#SBATCH --job-name=qwen-lora-eval
#SBATCH --output=logs/qwen_lora_eval_%j.out
#SBATCH --error=logs/qwen_lora_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib

# Activate your zsh/conda environment
source ~/.zshrc
conda activate qwen25vl

# Move to the qwen-vl-finetune directory
cd /mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune

# ====== Qwen2.5-VL LoRA Model Evaluation ======

# Model and data paths
MODEL_PATH="../output/lora_video_action"
TEST_DATA="../annotations/qwen_annotations_original_videos/test.json"
IMAGE_FOLDER="../data"
OUTPUT_DIR="../output/lora_video_action/eval_results"
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

# Run evaluation (LoRA model)
echo "🚀 Starting evaluation with LoRA model..."
python scripts/eval_lora_video.py \
    --model_path $MODEL_PATH \
    --test_data $TEST_DATA \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --base_model $BASE_MODEL

# Uncomment the following lines to evaluate with base model instead
# echo "🚀 Starting evaluation with base model..."
# python scripts/eval_lora_video.py \
#     --model_path $MODEL_PATH \
#     --test_data $TEST_DATA \
#     --image_folder $IMAGE_FOLDER \
#     --output_dir $OUTPUT_DIR \
#     --base_model $BASE_MODEL \
#     --use_base_model

echo "✅ Evaluation complete!" 