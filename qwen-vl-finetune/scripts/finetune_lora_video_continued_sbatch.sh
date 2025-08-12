#!/bin/bash
#SBATCH --job-name=lora_video_continued
#SBATCH --output=logs/lora_video_continued_%j.out
#SBATCH --error=logs/lora_video_continued_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --nodelist=oprime.ib

source ~/.bashrc
source ~/.zshrc

# Initialize conda properly for SLURM
eval "$(conda shell.bash hook)"
conda activate qwen25vl

# Install required dependencies
pip install --quiet -r requirements_training.txt

cd /mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune

# Set correct PYTHONPATH for this project only (override to avoid conflicts)
export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src

# Training parameters
DATA_PATH="../annotations/qwen_annotations_original_videos/train.json"  # Path to your LLaVA-style JSON
EVAL_DATA_PATH="../annotations/qwen_annotations_original_videos/val.json"  # Path to validation JSON
IMAGE_FOLDER="../data"                                 # Root folder containing original video_*/ (not qwen_data)
OUTPUT_DIR="./output/lora_video_action_continued"     # Output directory for continued training
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"            # Base model to fine-tune

# Training hyperparameters
EPOCHS=12                                            # Total epochs (6 more from checkpoint 288)
BATCH_SIZE=2                                         # Per-device batch size (optimized for 8x A6000)
GRAD_ACCUM_STEPS=4                                   # Gradient accumulation steps
LEARNING_RATE=1e-4                                   # Learning rate (matching original)
WARMUP_RATIO=0.03                                    # Warmup ratio (matching original)

# Validation settings
EVAL_STRATEGY="steps"
EVAL_STEPS=100                                       # Evaluate every 100 steps (~1.25 epochs)
EVAL_ACCUMULATION_STEPS=4                            # Validation accumulation steps

# LoRA settings (matching original training)
LORA_RANK=64
LORA_ALPHA=64
LORA_DROPOUT=0.05

# Video resolution and sampling
VIDEO_MIN_PIXELS=$((128 * 28 * 28))
VIDEO_MAX_PIXELS=$((768 * 28 * 28))
FPS=1.0

# Learning rates (matching original training)
LEARNING_RATE=1e-4
VISION_LR=2e-6
MERGER_LR=1e-5

# Resume from checkpoint
RESUME_CHECKPOINT="./output/lora_video_action/checkpoint-288"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "🚀 Starting continued LoRA training from checkpoint 288..."
echo "📊 Training for ${EPOCHS} total epochs (6 more epochs)"
echo "📈 Validation every ${EVAL_STEPS} steps"
echo "💾 Output directory: $OUTPUT_DIR"

# Run training with DeepSpeed (using the correct script)
deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_lora_modules -1 \
    --deepspeed ./scripts/zero3_offload.json \
    --model_id $BASE_MODEL \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --eval_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --eval_accumulation_steps $EVAL_ACCUMULATION_STEPS \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    --fps $FPS \
    --learning_rate $LEARNING_RATE \
    --merger_lr $MERGER_LR \
    --vision_lr $VISION_LR \
    --weight_decay 0.1 \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --resume_from_checkpoint $RESUME_CHECKPOINT

echo "✅ Continued training completed!"
echo "📊 Final model saved to: $OUTPUT_DIR"
echo "📈 Validation metrics logged to TensorBoard" 