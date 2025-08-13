# 🧠 Qwen2.5-VL Parkinson's Disease Video Analysis Project

## 📖 Table of Contents

- [📋 Overview](#-overview)
- [🏗️ Project Structure](#️-project-structure)
- [📁 File Layout](#-file-layout)
- [📁 Data Folder Setup](#-data-folder-setup)
  - [🔧 Required Data Structure](#-required-data-structure)
  - [📋 Module-Specific Requirements](#-module-specific-requirements)
  - [📋 Dataset Analysis](#-dataset-analysis)
  - [🔄 Annotation Format Differences](#-annotation-format-differences)
  - [⚠️ Critical Setup Notes](#️-critical-setup-notes)
- [🚀 Commands for Each Module](#-commands-for-each-module)
  - [🔍 Module 1: Zero-Shot Evaluation](#-module-1-zero-shot-evaluation)
  - [🚀 Module 2: LoRA Fine-Tuning & Evaluation](#-module-2-lora-fine-tuning--evaluation)
  - [📊 PD Insighter Format Output](#-pd-insighter-format-output)
  - [🌐 Module 3: Interactive Web Application](#-module-3-interactive-web-application)
- [🎯 Quick Start](#-quick-start)
- [📚 Additional Resources](#-additional-resources)

## 📋 Overview

This repository is based on **Qwen2.5-VL Pro** and extends it for Parkinson's disease video analysis. For basic installation and setup, please refer to the original [Qwen2.5-VL README.md](README.md) first.

## 🏗️ Project Structure

We have developed **3 main modules** for comprehensive video analysis:

### 🔍 Module 1: Zero-Shot Evaluation
Evaluates pre-trained Qwen2.5-VL models (7B/72B) on video action classification without fine-tuning. Provides baseline performance metrics and video annotation capabilities.

### 🚀 Module 2: LoRA Fine-Tuning & Evaluation  
Fine-tunes Qwen2.5-VL models using LoRA (Low-Rank Adaptation) on Parkinson's video data, then evaluates the trained models for improved performance on domain-specific tasks.

### 🌐 Module 3: Interactive Web Application
Provides an intuitive web interface for real-time video analysis, allowing users to upload videos, view analysis results, and interact with both zero-shot and fine-tuned models.

## 📁 File Layout

```
Qwen2.5-VL/
├── 📄 README_PARKINSON.md           # This file
├── 📄 REORGANIZATION_COMPLETE.md    # Project reorganization details
├── 📁 scripts/                      # 🎯 Main Entry Points
│   ├── train.sh                     # Training launcher
│   ├── evaluate.sh                  # Evaluation launcher
│   ├── web_app.py                   # Web app launcher
│   ├── setup.sh                     # Environment setup
│   └── requirements.txt             # Dependencies
├── 🧠 parkinson_proj/               # 🎯 Core Project Modules
│   ├── 🔍 evaluation/               # Module 1: Zero-Shot Evaluation
│   │   ├── zero_shot/               # Zero-shot analysis scripts
│   │   │   ├── analyze_video_actions.py     # Main video analysis
│   │   │   ├── check_video_action_accuracy.py  # Accuracy calculation
│   │   │   └── test_qwen25vl_fixed.py       # Model testing
│   │   ├── visualization/           # Result visualization
│   │   │   └── visualize_video_0_with_annotations.py
│   │   └── evaluation_results/      # Output results
│   ├── 🚀 training/                 # Module 2: LoRA Training
│   │   ├── 7b/                      # 7B model training scripts
│   │   │   ├── finetune_lora_video.sh
│   │   │   ├── finetune_lora_video_continued_sbatch.sh
│   │   │   └── merge_7b_lora.sh
│   │   ├── 72b/                     # 72B model training scripts
│   │   │   ├── finetune_lora_video_sbatch.sh
│   │   │   ├── finetune_lora_video_72b_continued_sbatch.sh
│   │   │   └── merge_72b_lora.sh
│   │   ├── configs/                 # Training configurations
│   │   │   ├── zero2.json, zero3.json, zero3_offload.json
│   │   └── evaluation/              # Fine-tuned model evaluation
│   │       ├── fine_tuned/          # LoRA evaluation scripts
│   │       │   ├── eval_lora_video.py
│   │       │   ├── eval_lora_video_corrected.py
│   │       │   └── eval_lora_video_sbatch.sh
│   │       └── zero_shot/           # Baseline evaluation
│   ├── 🌐 web_application/          # Module 3: Web Interface
│   │   ├── core/                    # Core analysis engine
│   │   ├── web_interface/           # Gradio web app
│   │   │   ├── app.py               # Main web application
│   │   │   └── launch_web.py        # Web launcher
│   │   ├── cli/                     # Command-line interface
│   │   │   └── cli_app.py
│   │   ├── config/                  # Configuration files
│   │   ├── utils/                   # Utility functions
│   │   └── prompts/                 # AI prompts
│   └── 📋 main_scripts/             # Project launchers
│       ├── train_models.sh
│       ├── evaluate_models.sh
│       └── launch_web_app.sh
├── 📁 data/                         # Video datasets
├── 📁 annotations/                  # Training annotations
├── 📁 output/                       # Model outputs
└── 📁 to_be_cleaned/               # Old files (can be deleted)
```

## 📁 Data Folder Setup

### 🔧 **Required Data Structure**

After downloading your Parkinson's disease video dataset, organize your data folder as follows:

```
Qwen2.5-VL/
├── 📁 data/                            # 🎯 MAIN VIDEO DATA
│   ├── video_0/                        # ✅ PRIMARY - Used by all modules (535 val/test samples)
│   │   ├── clip_000.mp4               # Individual video clips
│   │   ├── clip_001.mp4               # Supported: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm
│   │   ├── clip_002.mp4               
│   │   └── ...                        # (1000+ clips typically)
│   ├── video_1/                        # 🚀 REQUIRED for training (909 samples)
│   ├── video_2/                        # 🚀 REQUIRED for training (469 samples)  
│   ├── video_3/                        # 🚀 REQUIRED for training (463 samples)
│   ├── video_4/                        # 🚀 REQUIRED for training (316 samples)
│   ├── video_5/                        # 🚀 REQUIRED for training (922 samples)
│   └── qwen_data/                      # ❌ DEPRECATED: Contains damaged videos (not used)
├── 📁 annotations/                     # 🏷️ REQUIRED FOR TRAINING
│   ├── mode2/                          # 📊 Original CSV annotations
│   │   ├── train.csv                  # Training labels (3,079 samples)
│   │   ├── val.csv                    # Validation labels (535 samples)  
│   │   └── test.csv                   # Test labels (535 samples)
│   ├── qwen_annotations/               # ❌ DEPRECATED: Points to damaged qwen_data/ (not used)
│   │   ├── train.json                 
│   │   ├── val.json                   
│   │   └── test.json                  
│   └── qwen_annotations_original_videos/ # 🚀 TRAINING FORMAT: Used by both 7B & 72B models
│       ├── train.json                 # Points to video_1/, video_2/, video_3/, video_4/, video_5/
│       ├── val.json                   # Points to video_0/ (535 samples)
│       └── test.json                  # Points to video_0/ (535 samples)
└── 📁 output/                          # 💾 AUTO-CREATED: Model outputs
    ├── lora_video_action_7b_merged/    # 7B fine-tuned models
    └── lora_video_action_72b_merged/   # 72B fine-tuned models
```

### 📋 **Module-Specific Requirements**

#### 🔍 **Module 1: Zero-Shot Evaluation**
**Required folders:**
```
data/
└── video_0/           # ✅ ESSENTIAL: Default video folder for analysis
    ├── clip_000.mp4   # Individual video clips for testing
    └── ...            
```

#### 🚀 **Module 2: LoRA Fine-Tuning**

**For Both 7B & 72B Model Training:**
```
data/
├── video_0/           # ✅ REQUIRED: 535 val/test samples  
├── video_1/           # ✅ REQUIRED: 909 training samples
├── video_2/           # ✅ REQUIRED: 469 training samples
├── video_3/           # ✅ REQUIRED: 463 training samples
├── video_4/           # ✅ REQUIRED: 316 training samples
└── video_5/           # ✅ REQUIRED: 922 training samples
annotations/
└── qwen_annotations_original_videos/  # Used by both 7B & 72B models
    ├── train.json     # Uses video_1, video_2, video_3, video_4, video_5
    ├── val.json       # Uses video_0 
    └── test.json      # Uses video_0
```

#### 🌐 **Module 3: Web Application**
**Required folders:**
```
data/
└── video_0/           # ✅ ESSENTIAL: Default video folder for web interface
    ├── clip_000.mp4   # Individual video clips for analysis
    └── ...            
parkinson_proj/web_application/output/  # 🔄 AUTO-CREATED: Analysis exports
```

### 📋 **Dataset Analysis**

**Action Label Distribution (from annotations/mode2/train.csv):**
- **sitting**: 1,663 samples (54%)
- **walking**: 704 samples (23%)  
- **standing**: 690 samples (22%)
- **upstair**: 12 samples (0.4%)
- **downstair**: 10 samples (0.3%)

**Video Source Distribution:**
- **video_1**: 909 samples
- **video_5**: 922 samples  
- **video_2**: 469 samples
- **video_3**: 463 samples
- **video_4**: 316 samples

### 🔄 **Annotation Format Differences**

**Two annotation formats are provided:**

1. **`qwen_annotations/`** - ❌ **DEPRECATED** (points to damaged qwen_data/):
   ```json
   {
     "video": "qwen_data/video_0/clip_504.mp4",  // Points to damaged processed data
     "conversations": [{"from": "human", "value": "<video>\nWhat action is being performed?"}]
   }
   ```

2. **`qwen_annotations_original_videos/`** - ✅ **CURRENT** (used by both 7B & 72B):
   ```json
   {
     "video": "video_1/clip_504.mp4",  // Points to original video folders
     "conversations": [{"from": "human", "value": "<video>\nWhat action is being performed?"}]
   }
   ```

### ⚠️ **Critical Setup Notes**

#### **Minimum Requirements by Module:**
- **Module 1 (Zero-shot)**: Only needs `data/video_0/`
- **Module 2 (Both 7B & 72B Training)**: Needs `data/video_0/`, `video_1/`, `video_2/`, `video_3/`, `video_4/`, `video_5/` + `annotations/qwen_annotations_original_videos/`
- **Module 3 (Web App)**: Uses `data/video_0/` by default

#### **Key Changes:**
- **⚠️ IMPORTANT**: `data/qwen_data/` contains **damaged videos** and is no longer used
- **Both 7B & 72B training**: Now use original data split across `video_1/` through `video_5/` for training
- **Both models**: Use `video_0/` for validation and testing
- **Annotation format**: Both models use `annotations/qwen_annotations_original_videos/`
- The `output/` folder is automatically created during training
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`

#### **Missing Folders Impact:**
- Missing `video_1/` through `video_5/`: **Both 7B & 72B training will fail**
- Missing `video_0/`: **All modules will fail** 
- Missing `qwen_data/`: No impact (deprecated folder with damaged videos)
- Missing annotations: Training will **fail**, zero-shot/web app still work

## 🚀 Commands for Each Module

### 🔍 Module 1: Zero-Shot Evaluation

#### Zero-Shot Analysis (7B Model)
```bash
# Analyze videos with 7B model
python parkinson_proj/evaluation/zero_shot/analyze_video_actions.py \
    --video_folder "data/video_0" \
    --output_file "parkinson_proj/evaluation/evaluation_results/zero_shot_7b_results.json" \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct"
```
**📁 Output**: `parkinson_proj/evaluation/evaluation_results/zero_shot_7b_results.json`

#### Zero-Shot Analysis (72B Model)  
```bash
# Analyze videos with 72B model
python parkinson_proj/evaluation/zero_shot/analyze_video_actions.py \
    --video_folder "data/video_0" \
    --output_file "parkinson_proj/evaluation/evaluation_results/zero_shot_72b_results.json" \
    --model_id "Qwen/Qwen2.5-VL-72B-Instruct"
```
**📁 Output**: `parkinson_proj/evaluation/evaluation_results/zero_shot_72b_results.json`

#### Calculate Accuracy (7B Model)
```bash
# Calculate accuracy for 7B zero-shot results
python parkinson_proj/evaluation/zero_shot/check_video_action_accuracy.py \
    --results_file "parkinson_proj/evaluation/evaluation_results/zero_shot_7b_results.json" \
    --ground_truth "annotations/ground_truth.json"
```
**📁 Output**: Console output with accuracy metrics

#### Calculate Accuracy (72B Model)
```bash
# Calculate accuracy for 72B zero-shot results  
python parkinson_proj/evaluation/zero_shot/check_video_action_accuracy.py \
    --results_file "parkinson_proj/evaluation/evaluation_results/zero_shot_72b_results.json" \
    --ground_truth "annotations/ground_truth.json"
```
**📁 Output**: Console output with accuracy metrics

#### Video Annotation
```bash
# Generate annotated video with results
python parkinson_proj/evaluation/visualization/visualize_video_0_with_annotations.py \
    --video_path "data/video_0/video.mp4" \
    --results_file "parkinson_proj/evaluation/evaluation_results/zero_shot_results.json" \
    --output_path "output/annotated_video.mp4"
```
**📁 Output**: `output/annotated_video.mp4`

### 🚀 Module 2: LoRA Fine-Tuning & Evaluation

#### LoRA Fine-Tuning (7B Model)
```bash
# Recommended: Use reorganized script (identical functionality)
cd parkinson_proj/training/7b && bash finetune_lora_video.sh

# Alternative: Use original working script
bash qwen-vl-finetune/scripts/finetune_lora_video.sh
```
**📁 Output**: `output/lora_video_action_7b_merged/` (training checkpoints and final model)

#### LoRA Fine-Tuning (72B Model)
```bash  
# Recommended: Use reorganized script (identical functionality)
cd parkinson_proj/training/72b && sbatch finetune_lora_video_sbatch.sh

# Alternative: Use original working script
sbatch qwen-vl-finetune/scripts/finetune_lora_video_sbatch.sh
```
**📁 Output**: `output/lora_video_action_72b_merged/` (training checkpoints and final model)

#### Evaluate Fine-Tuned Model (7B)
```bash
# Recommended: Use reorganized script (fixed line continuation)
PYTHONPATH=./qwen-vl-utils/src python parkinson_proj/training/evaluation/fine_tuned/eval_lora_video_corrected.py \
  --model_path output/lora_video_action_7b_merged \
  --test_data annotations/qwen_annotations_original_videos/test.json \
  --val_data annotations/qwen_annotations_original_videos/val.json \
  --image_folder data \
  --use_merged_model

# Alternative: Use original script  
PYTHONPATH=./qwen-vl-utils/src python qwen-vl-finetune/scripts/eval_lora_video_corrected.py \
  --model_path output/lora_video_action_7b_merged \
  --test_data annotations/qwen_annotations_original_videos/test.json \
  --val_data annotations/qwen_annotations_original_videos/val.json \
  --image_folder data \
  --use_merged_model
```
**📁 Output**: `output/lora_video_action_7b_merged/eval_results/`
- `test_predictions.json` (detailed predictions)
- `pd_insighter_format_predictions.json` (🔗 **NEW**: PD insighter format for downstream analysis)
- `test_accuracy.json` (accuracy metrics)  
- `test_confusion_matrix.csv` (confusion matrix)
- `test_per_class_results.csv` (per-class results)

#### Evaluate Fine-Tuned Model (72B)
```bash
# Recommended: Use reorganized script (fixed line continuation)
PYTHONPATH=./qwen-vl-utils/src python parkinson_proj/training/evaluation/fine_tuned/eval_lora_video_corrected.py \
  --model_path output/lora_video_action_72b_merged \
  --test_data annotations/qwen_annotations_original_videos/test.json \
  --val_data annotations/qwen_annotations_original_videos/val.json \
  --image_folder data \
  --use_merged_model

# Alternative: Use original script
PYTHONPATH=./qwen-vl-utils/src python qwen-vl-finetune/scripts/eval_lora_video_corrected.py \
  --model_path output/lora_video_action_72b_merged \
  --test_data annotations/qwen_annotations_original_videos/test.json \
  --val_data annotations/qwen_annotations_original_videos/val.json \
  --image_folder data \
  --use_merged_model
```
**📁 Output**: `output/lora_video_action_72b_merged/eval_results/`
- `test_predictions.json` (detailed predictions)
- `pd_insighter_format_predictions.json` (🔗 **NEW**: PD insighter format for downstream analysis)
- `test_accuracy.json` (accuracy metrics)
- `test_confusion_matrix.csv` (confusion matrix)  
- `test_per_class_results.csv` (per-class results)

### 📊 **PD Insighter Format Output**

All evaluation scripts now automatically generate `pd_insighter_format_predictions.json` alongside the standard outputs. This format is designed for downstream analysis and integrates consecutive video clips into action segments.

**Format Structure:**
```json
{
  "sitting": [
    {
      "Starting frame": 306,
      "Ending frame": 339
    },
    {
      "Starting frame": 578,
      "Ending frame": 791
    }
  ],
  "walking": [
    {
      "Starting frame": 32,
      "Ending frame": 186
    },
    {
      "Starting frame": 339,
      "Ending frame": 405
    }
  ],
  "standing": [
    {
      "Starting frame": 0,
      "Ending frame": 32
    }
  ]
}
```

**Key Features:**
- **Sequential Processing**: Clips are sorted by number (`video_0_clip_340` → `clip_340`)
- **Action Grouping**: Consecutive clips with same predicted action are merged into segments
- **Frame Calculation**: Uses 30 FPS default (`clip_340` = frames 10200-10229)
- **Segment Detection**: Automatically detects action boundaries and transitions
- **Standard Format**: Matches expected downstream analysis pipeline requirements

**Technical Details:**
- **Input**: `test_predictions.json` with individual clip predictions
- **Processing**: Groups consecutive clips with identical predicted actions
- **Output**: Action segments with start/end frame numbers
- **Frame Rate**: Configurable (default: 30 FPS)
- **Gap Handling**: Non-consecutive clips create separate segments

This format enables seamless integration with PD analysis tools and provides temporal action segmentation from clip-level predictions.

### 🌐 Module 3: Interactive Web Application

#### Launch Web Application
```bash
# Recommended: Use reorganized web app
python parkinson_proj/web_application/web_interface/app.py

# Alternative: Use project launcher
bash parkinson_proj/main_scripts/launch_web_app.sh
```
**📁 Output**: Web interface accessible at `http://localhost:7860`

#### Web Application Interaction Steps

1. **🚀 Start the Web App**
   - Run the launch command above
   - Open browser to `http://localhost:7860`

2. **📹 Upload Video**
   - Click "Upload Video" button
   - Select your Parkinson's video file (.mp4, .avi, etc.)
   - Wait for video processing

3. **⚙️ Configure Analysis**
   - Choose model type (Zero-shot vs Fine-tuned)
   - Select model size (7B vs 72B)
   - Adjust analysis parameters if needed

4. **▶️ Run Analysis**
   - Click "Analyze Video" button
   - Monitor progress in real-time
   - View segment-by-segment results

5. **📊 Review Results**
   - Browse action classifications
   - Check confidence scores
   - View suspicious segments
   - Export results as JSON

6. **💬 Interactive Query**
   - Ask questions: "What clips look wrong?"
   - Get detailed explanations
   - Request medical analysis
   - Generate reports

**📁 Analysis Output**: Results exported to `parkinson_proj/web_application/output/analysis_export_YYYYMMDD_HHMMSS.json`

## 🎯 Quick Start

```bash
# 1. Setup environment
./scripts/setup.sh --env-name qwen25vl
conda activate qwen25vl

# 2. Run zero-shot evaluation
./scripts/evaluate.sh --type zero-shot --data data/video_0 --output results/

# 3. Train a model
./scripts/train.sh --model 7b --mode fresh

# 4. Launch web interface
python scripts/web_app.py --interface web
```

## 📚 Additional Resources

- **Original Qwen2.5-VL**: See main [README.md](README.md)
- **Setup Guide**: Run `./scripts/setup.sh --help`
- **Training Guide**: Run `./scripts/train.sh --help`  
- **Evaluation Guide**: Run `./scripts/evaluate.sh --help`
- **Web App Guide**: Run `python scripts/web_app.py --help`

For detailed technical documentation, check the module-specific README files in each directory.