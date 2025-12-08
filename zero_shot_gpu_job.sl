#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH -t 2-00:00:00          # 2 days (full format recommended)
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen_eval
#SBATCH --output=logs/qwen_eval_%j.out
#SBATCH --error=logs/qwen_eval_%j.err

# ----------------------------
# Load modules and environment
# ----------------------------
module purge
module load anaconda/2024.02  # Load Anaconda after purge

# Activate your virtual environment
source ~/venv/bin/activate

# (Optional but helpful) Confirm GPU access
nvidia-smi

# ----------------------------
# Run your evaluation script
# ----------------------------
python parkinson_proj/evaluation/zero_shot/analyze_video_actions.py \
    --video_folder "data/video_j" \
    --output_file "parkinson_proj/evaluation/evaluation_results/zero_shot_7b_results.json" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct"
