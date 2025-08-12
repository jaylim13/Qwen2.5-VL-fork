#!/bin/bash

# ====== LoRA Weight Merging Script ======

# Configuration
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH="../output/lora_video_action"
MERGED_MODEL_PATH="../output/lora_video_action_merged"

echo "🔄 Starting LoRA weight merging..."
echo "Base model: $BASE_MODEL"
echo "LoRA path: $LORA_PATH"
echo "Merged model path: $MERGED_MODEL_PATH"

# Activate environment
conda activate qwen25vl

# Set Python path
export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

# Run merging
python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

print('🔄 Loading base model...')
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '$BASE_MODEL',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

print('🔄 Loading LoRA adapter...')
model = PeftModel.from_pretrained(base_model, '$LORA_PATH')

print('🔄 Merging LoRA weights...')
model = model.merge_and_unload()

print('💾 Saving merged model...')
model.save_pretrained('$MERGED_MODEL_PATH')
processor = AutoProcessor.from_pretrained('$BASE_MODEL')
processor.save_pretrained('$MERGED_MODEL_PATH')

print('✅ LoRA merging complete!')
print(f'📁 Merged model saved to: $MERGED_MODEL_PATH')
"