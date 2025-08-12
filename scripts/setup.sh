#!/bin/bash
# Setup script for Qwen2.5-VL Parkinson's Project

set -e

# Print usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup script for Qwen2.5-VL Parkinson's Project"
    echo ""
    echo "Options:"
    echo "  --env-name     Name for conda environment [default: qwen25vl]"
    echo "  --skip-conda   Skip conda environment creation"
    echo "  --help         Show this help message"
    echo ""
}

# Parse command line arguments
ENV_NAME="qwen25vl"
SKIP_CONDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --skip-conda)
            SKIP_CONDA=true
            shift
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

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up Qwen2.5-VL Parkinson's Project"
echo "=" * 50
echo "📁 Project Root: $PROJECT_ROOT"
echo "🐍 Environment Name: $ENV_NAME"
echo "=" * 50

# Navigate to project root
cd "$PROJECT_ROOT"

# Create conda environment if not skipped
if [[ "$SKIP_CONDA" == false ]]; then
    echo "🔄 Creating conda environment: $ENV_NAME"
    
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "⚠️  Environment $ENV_NAME already exists. Skipping creation."
    else
        # Check if environment.yaml exists
        if [[ -f "qwen-vl-finetune/environment.yaml" ]]; then
            echo "📋 Using environment.yaml for setup"
            conda env create -f qwen-vl-finetune/environment.yaml -n "$ENV_NAME"
        else
            echo "📋 Creating basic environment"
            conda create -n "$ENV_NAME" python=3.10 -y
        fi
    fi
    
    echo "✅ Conda environment ready"
    echo ""
    echo "🔄 To activate the environment, run:"
    echo "   conda activate $ENV_NAME"
    echo ""
else
    echo "⏭️  Skipping conda environment creation"
fi

# Install requirements
echo "📦 Installing requirements..."

# Check if requirements files exist and install them
REQUIREMENTS_FILES=(
    "qwen-vl-finetune/requirements.txt"
    "qwen-vl-finetune/requirements_training.txt"
    "requirements_web_demo.txt"
)

for req_file in "${REQUIREMENTS_FILES[@]}"; do
    if [[ -f "$req_file" ]]; then
        echo "📋 Installing from $req_file"
        pip install -r "$req_file"
    else
        echo "⚠️  Requirements file not found: $req_file"
    fi
done

# Set up directories
echo "🗂️  Setting up directories..."

# Create necessary directories if they don't exist
DIRECTORIES=(
    "data"
    "annotations"
    "output"
    "logs"
    "parkinson_proj/evaluation/evaluation_results"
)

for dir in "${DIRECTORIES[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "📁 Creating directory: $dir"
        mkdir -p "$dir"
    else
        echo "✅ Directory exists: $dir"
    fi
done

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Set up Python path
echo "🐍 Setting up Python path..."
echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"" > setup_env.sh
echo "echo '🧠 Qwen2.5-VL Parkinson Project environment ready!'" >> setup_env.sh
chmod +x setup_env.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Quick Start Guide:"
echo "=" * 50
echo "1. Activate environment:    conda activate $ENV_NAME"
echo "2. Set up environment:      source setup_env.sh"
echo "3. Start training:          ./scripts/train.sh --help"
echo "4. Run evaluation:          ./scripts/evaluate.sh --help"
echo "5. Launch web app:          python scripts/web_app.py --help"
echo ""
echo "📚 For more information, check the README files in each module."