#!/bin/bash
#SBATCH --job-name=qwen-lora-video-72b-continued
#SBATCH --output=logs/qwen_lora_video_72b_continued_%j.out
#SBATCH --error=logs/qwen_lora_video_72b_continued_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --partition=a6000
#SBATCH --nodelist=oprime.ib

# Activate your zsh/conda environment
source ~/.zshrc

# Initialize conda properly for SLURM
eval "$(conda shell.bash hook)"
conda activate qwen25vl

# Fix DeepSpeed async I/O issue on oprime
export TMPDIR=/tmp
export DS_BUILD_ASYNC_IO=0

# Move to the qwen-vl-finetune directory
cd /mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune

# ====== Qwen2-VL LoRA Finetuning for Video Action Classification (Continued) ======

# Model selection (choose your base model)
MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"

# Data paths (edit these as needed)
DATA_PATH="../annotations/qwen_annotations_original_videos/train.json"  # Path to your LLaVA-style JSON
EVAL_DATA_PATH="../annotations/qwen_annotations_original_videos/val.json"  # Path to validation JSON
IMAGE_FOLDER="../data"                                 # Root folder containing original video_*/ (not qwen_data)

# Output
OUTPUT_DIR="../output/lora_video_action_72b"

# Training configuration (optimized for 8x A6000 with 72B model)
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
EPOCHS=10

# Validation configuration
EVAL_STRATEGY="steps"
EVAL_STEPS=200
EVAL_ACCUMULATION_STEPS=4

# LoRA configuration (adjusted for 72B model)
LORA_RANK=128
LORA_ALPHA=128
LORA_DROPOUT=0.1
NUM_LORA_MODULES=-1

# Video resolution and sampling
VIDEO_MIN_PIXELS=$((128 * 28 * 28))
VIDEO_MAX_PIXELS=$((768 * 28 * 28))
FPS=1.0

# Learning rates (adjusted for 72B model)
LEARNING_RATE=5e-5
VISION_LR=1e-6
MERGER_LR=5e-6

# Deepspeed config
DEEPSPEED_CONFIG="scripts/zero3_offload.json"

# Resume from checkpoint (will be set when checkpoint-200 is available)
RESUME_CHECKPOINT="./output/lora_video_action_72b/checkpoint-200"

export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

# Install required dependencies
pip install --quiet -r requirements_training.txt

echo "🚀 Starting continued LoRA training from checkpoint 200..."
echo "📊 Training for $EPOCHS total epochs"
echo "📈 Validation every $EVAL_STEPS steps"
echo "💾 Output directory: $OUTPUT_DIR"

# Run training (relative path to train_sft.py)
deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_lora_modules $NUM_LORA_MODULES \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_id $MODEL_NAME \
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
    --per_device_train_batch_size $BATCH_PER_DEVICE \
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
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --resume_from_checkpoint $RESUME_CHECKPOINT 