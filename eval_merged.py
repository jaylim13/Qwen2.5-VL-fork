from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load the ORIGINAL Qwen base model (not your fine-tuned one)
base_model_id = "jaylim3/qwen-2.5"  # ← Changed to original base
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Original Qwen base model loaded")

# Load LoRA adapter from your HuggingFace repo
adapter_repo = "jaylim3/qwen-2.5"  # Your repo with the adapter
model = PeftModel.from_pretrained(
    model, 
    adapter_repo,
    subfolder="final_model"
)

print("✅ Adapter loaded from your repo")

# Merge
merged_model = model.merge_and_unload()
print("✅ Model merged")

# Save the merged model locally
merged_output_dir = "./merged_qwen_video_model"
merged_model.save_pretrained(merged_output_dir)

# Save processor (from the original base)
processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
processor.save_pretrained(merged_output_dir)

print(f"✅ Merged model saved to {merged_output_dir}")
print("This model = Original Qwen + Your action classification fine-tuning")