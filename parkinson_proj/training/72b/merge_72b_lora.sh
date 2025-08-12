#!/bin/zsh

# Merge 72B LoRA weights with base model
echo "🔄 Merging 72B LoRA weights (checkpoint-50)..."

# Set paths
LORA_PATH="output/lora_video_action_72b/checkpoint-50"
BASE_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
OUTPUT_PATH="output/lora_video_action_72b_merged"

# Create output directory
mkdir -p $OUTPUT_PATH

# Activate environment (use qwen25vl like the working script)
source ~/.zshrc
conda activate qwen25vl

# Set Python path
export PYTHONPATH=/mnt/arc/yygx/pkgs_baselines/Qwen2.5-VL/qwen-vl-finetune:src:$PYTHONPATH

# Run merging (using the same approach as merge_lora_latest.sh)
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
merged_model = model.merge_and_unload()

print('🔄 Saving merged model...')
merged_model.save_pretrained('$OUTPUT_PATH')
processor = AutoProcessor.from_pretrained('$BASE_MODEL')
processor.save_pretrained('$OUTPUT_PATH')

print('✅ 72B LoRA merging complete!')
print(f'📁 Merged model saved to: $OUTPUT_PATH')
"

echo "✅ 72B model merged to: $OUTPUT_PATH" 