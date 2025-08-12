#!/bin/bash

# ====== Qwen2-VL LoRA Finetuning for Video Action Classification ======

# Activate your zsh/conda environment (uncomment if running interactively)
# source ~/.zshrc
# conda activate qwen25vl

# Move to the qwen-vl-finetune directory (required for training)
cd /mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune

# Model selection (choose your base model)
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Data paths (edit these as needed)
DATA_PATH="../annotations/qwen_annotations/train.json"  # Path to your LLaVA-style JSON
IMAGE_FOLDER="../data"                                 # Root folder containing video_*/

# Output
OUTPUT_DIR="../output/lora_video_action_7b_merged"

# Training configuration
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
EPOCHS=3

# LoRA configuration
LORA_RANK=64
LORA_ALPHA=64
LORA_DROPOUT=0.05
NUM_LORA_MODULES=-1

# Video resolution and sampling
VIDEO_MIN_PIXELS=$((128 * 28 * 28))
VIDEO_MAX_PIXELS=$((768 * 28 * 28))
FPS=1.0

# Learning rates
LEARNING_RATE=1e-4
VISION_LR=2e-6
MERGER_LR=1e-5

# Deepspeed config
DEEPSPEED_CONFIG="scripts/zero3_offload.json"

export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

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
    --resume_from_checkpoint False 