#!/bin/bash
# Main evaluation entry point script for Qwen2.5-VL Parkinson's Project

set -e

# Print usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Main evaluation script for Qwen2.5-VL Parkinson's Project"
    echo ""
    echo "Options:"
    echo "  --type         Evaluation type (zero-shot|fine-tuned) [default: zero-shot]"
    echo "  --model        Model path for fine-tuned evaluation"
    echo "  --data         Test data path"
    echo "  --output       Output directory for results"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --type zero-shot"
    echo "  $0 --type fine-tuned --model ../output/lora_video_action"
    echo ""
}

# Parse command line arguments
EVAL_TYPE="zero-shot"
MODEL_PATH=""
DATA_PATH=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
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
if [[ "$EVAL_TYPE" != "zero-shot" && "$EVAL_TYPE" != "fine-tuned" ]]; then
    echo "❌ Error: Evaluation type must be 'zero-shot' or 'fine-tuned'"
    exit 1
fi

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Qwen2.5-VL Evaluation"
echo "=" * 50
echo "🔍 Evaluation Type: $EVAL_TYPE"
echo "📁 Project Root: $PROJECT_ROOT"
echo "=" * 50

# Navigate to appropriate evaluation directory
if [[ "$EVAL_TYPE" == "zero-shot" ]]; then
    EVAL_DIR="$PROJECT_ROOT/parkinson_proj/evaluation/zero_shot"
    
    # Set default values for zero-shot evaluation
    if [[ -z "$DATA_PATH" ]]; then
        DATA_PATH="$PROJECT_ROOT/data"
    fi
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$PROJECT_ROOT/parkinson_proj/evaluation/evaluation_results"
    fi
    
    cd "$EVAL_DIR"
    
    echo "📋 Running zero-shot video action analysis..."
    python analyze_video_actions.py \
        --video_folder "$DATA_PATH" \
        --output_file "$OUTPUT_DIR/zero_shot_results.json"
    
else  # fine-tuned
    EVAL_DIR="$PROJECT_ROOT/parkinson_proj/training/evaluation/fine_tuned"
    
    if [[ -z "$MODEL_PATH" ]]; then
        echo "❌ Error: Model path is required for fine-tuned evaluation"
        echo "Use --model to specify the path to the trained model"
        exit 1
    fi
    
    # Set default values for fine-tuned evaluation
    if [[ -z "$DATA_PATH" ]]; then
        DATA_PATH="$PROJECT_ROOT/annotations/qwen_annotations/test.json"
    fi
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$(dirname "$MODEL_PATH")/eval_results"
    fi
    
    cd "$EVAL_DIR"
    
    echo "📋 Running fine-tuned model evaluation..."
    echo "🤖 Model: $MODEL_PATH"
    echo "📊 Test Data: $DATA_PATH"
    echo "💾 Output: $OUTPUT_DIR"
    
    python eval_lora_video.py \
        --model_path "$MODEL_PATH" \
        --test_data "$DATA_PATH" \
        --image_folder "$PROJECT_ROOT/data" \
        --output_dir "$OUTPUT_DIR"
fi

echo "✅ Evaluation completed successfully!"
echo "📊 Results saved to: $OUTPUT_DIR"