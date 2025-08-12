#!/bin/bash
#SBATCH --job-name=merge_lora_latest
#SBATCH --output=logs/merge_lora_latest_%j.out
#SBATCH --error=logs/merge_lora_latest_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/.bashrc
conda activate qwen25vl

# Configuration
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH="./output/lora_video_action"
OUTPUT_PATH="./output/lora_video_action_merged_latest"

echo "🔄 Merging LoRA weights (latest checkpoint) into standalone model..."
echo "📁 LoRA path: $LORA_PATH"
echo "📁 Output path: $OUTPUT_PATH"

# Create output directory
mkdir -p $OUTPUT_PATH

# Merge LoRA weights into base model
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

print('✅ Merged model saved to: $OUTPUT_PATH')
print('📊 Model size: ~15GB (full standalone model)')
"

echo "✅ Merge completed! New merged model at: $OUTPUT_PATH" 