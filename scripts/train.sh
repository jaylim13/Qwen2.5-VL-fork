#!/bin/bash
# Main training entry point script for Qwen2.5-VL Parkinson's Project

set -e

# Print usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Main training script for Qwen2.5-VL Parkinson's Project"
    echo ""
    echo "Options:"
    echo "  --model        Model size to train (7b|72b) [default: 7b]"
    echo "  --mode         Training mode (fresh|continue) [default: fresh]"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model 7b --mode fresh     # Start fresh 7B training"
    echo "  $0 --model 72b --mode continue # Continue 72B training"
    echo ""
}

# Parse command line arguments
MODEL_SIZE="7b"
TRAINING_MODE="fresh"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$MODEL_SIZE" != "7b" && "$MODEL_SIZE" != "72b" ]]; then
    echo "❌ Error: Model size must be '7b' or '72b'"
    exit 1
fi

if [[ "$TRAINING_MODE" != "fresh" && "$TRAINING_MODE" != "continue" ]]; then
    echo "❌ Error: Training mode must be 'fresh' or 'continue'"
    exit 1
fi

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Qwen2.5-VL Training"
echo "=" * 50
echo "📊 Model Size: $MODEL_SIZE"
echo "🔄 Training Mode: $TRAINING_MODE"
echo "📁 Project Root: $PROJECT_ROOT"
echo "=" * 50

# Navigate to training directory
TRAINING_DIR="$PROJECT_ROOT/parkinson_proj/training/$MODEL_SIZE"

if [[ ! -d "$TRAINING_DIR" ]]; then
    echo "❌ Error: Training directory not found: $TRAINING_DIR"
    exit 1
fi

cd "$TRAINING_DIR"

# Select appropriate script based on model size and mode
if [[ "$MODEL_SIZE" == "7b" ]]; then
    if [[ "$TRAINING_MODE" == "fresh" ]]; then
        SCRIPT_NAME="finetune_lora_video.sh"
    else
        SCRIPT_NAME="finetune_lora_video_continued_sbatch.sh"
    fi
else  # 72b
    if [[ "$TRAINING_MODE" == "fresh" ]]; then
        SCRIPT_NAME="finetune_lora_video_sbatch.sh"
    else
        SCRIPT_NAME="finetune_lora_video_72b_continued_sbatch.sh"
    fi
fi

TRAINING_SCRIPT="$TRAINING_DIR/$SCRIPT_NAME"

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "❌ Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "📋 Using training script: $SCRIPT_NAME"
echo "🔄 Starting training..."

# Execute the training script
bash "$TRAINING_SCRIPT"

echo "✅ Training script completed!"