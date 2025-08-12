# Qwen2.5-VL LoRA Video Action Classification - Evaluation & Statistics

This document describes the evaluation pipeline and dataset statistics tools for the Qwen2.5-VL LoRA video action classification project.

## 📁 File Structure

```
qwen-vl-finetune/scripts/
├── eval_lora_video.py              # Main evaluation script
├── eval_lora_video_sbatch.sh       # SLURM script for evaluation
└── README_evaluation.md            # This file

utils/
├── dataset_stats.py                # Dataset statistics script
└── run_dataset_stats.sh           # Shell script for statistics

output/lora_video_action/
├── eval_results/                   # Evaluation results
│   ├── test_predictions.json      # Raw model predictions
│   ├── test_accuracy.json         # Accuracy metrics
│   ├── test_confusion_matrix.csv  # Confusion matrix
│   └── test_per_class_results.csv # Per-class breakdown
└── dataset_stats/                  # Dataset statistics
    ├── dataset_summary.csv         # Overall statistics
    ├── label_distribution.csv      # Label distribution
    ├── video_source_distribution.csv # Video source distribution
    ├── detailed_statistics.json    # Detailed JSON stats
    └── dataset_balance_report.txt  # Balance analysis
```

## 🚀 Evaluation Pipeline

### 1. Model Evaluation Script

**File**: `eval_lora_video.py`

**Purpose**: Loads the trained LoRA model and evaluates it on the test set.

**Features**:
- Loads trained LoRA adapter from `output/lora_video_action/`
- Processes test videos with proper video handling
- Extracts action predictions from model responses
- Calculates comprehensive metrics (overall + per-class accuracy)
- Saves detailed results to `eval_results/` directory

**Usage**:
```bash
# Interactive mode
python scripts/eval_lora_video.py

# With custom parameters
python scripts/eval_lora_video.py \
    --model_path ../output/lora_video_action \
    --test_data ../annotations/qwen_annotations/test.json \
    --image_folder ../data \
    --output_dir ../output/lora_video_action/eval_results \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct
```

**SLURM Usage**:
```bash
sbatch scripts/eval_lora_video_sbatch.sh
```

### 2. Training with Validation

**Modified Files**:
- `train_sft.py` - Added validation dataset support
- `sft_dataset.py` - Modified to include eval dataset
- `finetune_lora_video_sbatch.sh` - Added validation parameters

**New Training Parameters**:
```bash
--eval_strategy "steps"              # Evaluate every N steps
--eval_steps 200                     # Evaluation frequency
--eval_accumulation_steps 4          # Gradient accumulation for eval
--metric_for_best_model "eval_loss"  # Metric to track
--load_best_model_at_end True        # Save best model
--eval_data_path $EVAL_DATA_PATH     # Validation dataset path
```

**Usage**:
```bash
# Run training with validation
sbatch scripts/finetune_lora_video_sbatch.sh
```

## 📊 Dataset Statistics

### 1. Statistics Script

**File**: `utils/dataset_stats.py`

**Purpose**: Analyzes train/val/test datasets and generates comprehensive reports.

**Features**:
- Loads and analyzes all dataset splits
- Calculates label and video source distributions
- Generates balance analysis reports
- Saves multiple output formats (CSV, JSON, TXT)

**Usage**:
```bash
# Run statistics analysis
python utils/dataset_stats.py

# With custom parameters
python utils/dataset_stats.py \
    --train_data ../annotations/qwen_annotations/train.json \
    --val_data ../annotations/qwen_annotations/val.json \
    --test_data ../annotations/qwen_annotations/test.json \
    --output_dir ../output/lora_video_action/dataset_stats
```

**Shell Script Usage**:
```bash
bash utils/run_dataset_stats.sh
```

### 2. Output Files

**Dataset Summary** (`dataset_summary.csv`):
- Total samples per split
- Unique labels and video sources
- Most common labels and sources

**Label Distribution** (`label_distribution.csv`):
- Count and percentage for each label per split
- Cross-split comparison

**Video Source Distribution** (`video_source_distribution.csv`):
- Distribution across video sources (video_0, video_1, etc.)
- Per-split breakdown

**Detailed Statistics** (`detailed_statistics.json`):
- Complete dataset overview
- Detailed statistics for each split
- All labels and video sources

**Balance Report** (`dataset_balance_report.txt`):
- Balance assessment for each split
- Imbalance ratio calculations
- Recommendations for dataset improvement

## 📈 Metrics and Results

### Evaluation Metrics

1. **Overall Accuracy**: Total correct predictions / total samples
2. **Per-class Accuracy**: Accuracy for each action class
3. **Confusion Matrix**: Detailed prediction vs. ground truth
4. **Raw Predictions**: Model responses for manual inspection

### Statistics Metrics

1. **Label Distribution**: Count and percentage of each action
2. **Video Source Distribution**: Distribution across video sources
3. **Balance Assessment**: Imbalance ratios and recommendations
4. **Cross-split Comparison**: Train/val/test consistency

## 🔧 Configuration

### Evaluation Configuration

**Model Loading**:
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- LoRA adapter: Loaded from training output directory
- Device: Auto-detected (GPU/CPU)

**Video Processing**:
- Min pixels: 128 × 28 × 28
- Max pixels: 768 × 28 × 28
- FPS: 1.0 (from training config)

**Generation Parameters**:
- Max new tokens: 32
- Temperature: 0.1 (deterministic)
- Do sample: False

### Training Configuration

**Validation Setup**:
- Evaluation strategy: Steps-based
- Evaluation frequency: Every 200 steps
- Metric: eval_loss (minimize)
- Best model: Automatically saved

**Data Paths**:
- Training: `../annotations/qwen_annotations/train.json`
- Validation: `../annotations/qwen_annotations/val.json`
- Test: `../annotations/qwen_annotations/test.json`

## 🎯 Usage Examples

### 1. Quick Evaluation
```bash
# Run evaluation on trained model
sbatch scripts/eval_lora_video_sbatch.sh
```

### 2. Dataset Analysis
```bash
# Analyze dataset statistics
bash utils/run_dataset_stats.sh
```

### 3. Training with Validation
```bash
# Train with validation monitoring
sbatch scripts/finetune_lora_video_sbatch.sh
```

### 4. Custom Evaluation
```bash
# Evaluate with custom parameters
python scripts/eval_lora_video.py \
    --model_path /path/to/model \
    --test_data /path/to/test.json \
    --output_dir /path/to/results
```

## 📋 Output Interpretation

### Evaluation Results

1. **Overall Accuracy**: Should be > 0.7 for good performance
2. **Per-class Accuracy**: Check for class-specific issues
3. **Confusion Matrix**: Identify common misclassifications
4. **Raw Predictions**: Verify model response quality

### Statistics Results

1. **Label Balance**: Ratio < 2.0 indicates good balance
2. **Video Source Distribution**: Should be similar across splits
3. **Cross-split Consistency**: Train/val/test should have similar distributions

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Check if LoRA adapter exists in model path
   - Verify base model ID is correct
   - Ensure sufficient GPU memory

2. **Video Processing Errors**:
   - Verify video files exist in image_folder
   - Check video format compatibility
   - Ensure proper video path structure

3. **Memory Issues**:
   - Reduce batch size for evaluation
   - Use gradient accumulation
   - Enable CPU offloading

### Performance Tips

1. **Evaluation Speed**:
   - Use single GPU for evaluation
   - Process videos in batches
   - Enable lazy preprocessing

2. **Memory Optimization**:
   - Use BF16 precision
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO-3

## 📝 Notes

- Evaluation results are saved in `output/lora_video_action/eval_results/`
- Dataset statistics are saved in `output/lora_video_action/dataset_stats/`
- Training logs include validation metrics in TensorBoard
- Best model is automatically saved based on validation loss 