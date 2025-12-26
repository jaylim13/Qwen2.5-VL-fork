"""
Upload full Qwen2.5-VL model to Hugging Face without copying files locally.
"""

import os
from huggingface_hub import upload_file

# -------- CONFIG -------- #
MODEL_DIR = "/Users/jayden/Qwen2.5-VL/models/yue_model"  # path to your model folder
REPO_ID = "jaylim3/qwen-2.5"  # your Hugging Face repo
HF_TOKEN = os.getenv("HF_TOKEN")  # or paste your token here

# Only upload relevant file types
EXTENSIONS = [".safetensors", ".json", ".txt"]

# ------------------------ #

def main():
    for root, dirs, files in os.walk(MODEL_DIR):
        for file_name in files:
            if any(file_name.endswith(ext) for ext in EXTENSIONS):
                local_path = os.path.join(root, file_name)
                # Keep folder structure relative to MODEL_DIR
                path_in_repo = os.path.relpath(local_path, MODEL_DIR)
                
                print(f"Uploading {path_in_repo} ...")
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=REPO_ID,
                    repo_type="model",
                    token=HF_TOKEN
                )
    print("✅ Upload completed!")

if __name__ == "__main__":
    main()
