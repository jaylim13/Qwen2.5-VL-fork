# 🔬 Future Research Directions

## 📋 Overview

This document outlines potential research extensions for the Qwen2.5-VL Parkinson's Disease Video Analysis Project. These directions are designed to guide future researchers in expanding the capabilities and improving the performance of the current system.

## 🚀 Extension Opportunities

### 1. 📊 Multi-Modal Data Integration: Adding IMU Motion Data

**Current Limitation**: The VLM currently only processes video data, missing valuable motion sensor information.

**Research Opportunity**: Integrate IMU (Inertial Measurement Unit) motion data to enhance action classification accuracy.

#### 🔍 Files to Investigate for IMU Integration:

**Core Training Infrastructure:**
- `qwen-vl-finetune/src/dataset/sft_dataset.py:19` - Data loading pipeline
- `qwen-vl-finetune/src/dataset/data_utils.py` - Add `get_imu_info()` function
- `qwen-vl-finetune/src/train/train_sft.py` - Training loop modifications
- `qwen-vl-finetune/src/params.py` - Add IMU-related arguments

**Model Architecture:**
- `qwen-vl-finetune/src/train/monkey_patch_forward.py` - Forward pass modifications
- `qwen-vl-finetune/src/trainer/sft_trainer.py` - Trainer adaptations

**Data Format Extensions:**
- `annotations/qwen_annotations_original_videos/` - Add IMU data paths to JSON format
- `data/` - Create `imu_data/` subfolder structure

#### 📋 Implementation Roadmap:
1. **Extend data format** to include IMU file references
2. **Modify dataset loading** to synchronize video + IMU data
3. **Adapt model architecture** for multi-modal fusion
4. **Update training scripts** to handle additional modality
5. **Expand evaluation metrics** for multi-modal performance

### 2. 🎯 Extended Action Classification

**Current Limitation**: Only 5 basic actions (sitting, walking, standing, upstair, downstair).

**Research Opportunity**: Expand to comprehensive Parkinson's-specific actions.

#### 📋 Current Action Distribution:
- **sitting**: 1,663 samples (54%)
- **walking**: 704 samples (23%)  
- **standing**: 690 samples (22%)
- **upstair**: 12 samples (0.4%)
- **downstair**: 10 samples (0.3%)

#### 🎯 Proposed Additional Actions:
- **freezing** - Sudden movement cessation
- **stumbling** - Balance loss events
- **tremor_episodes** - Visible tremor periods
- **bradykinesia** - Slow movement phases
- **dyskinesia** - Involuntary movements
- **turn_hesitation** - Difficulty initiating turns
- **gait_festination** - Rapid, shuffling steps

#### 🔧 Files to Modify for Action Extension:

**Annotation Files:**
- `annotations/mode2/*.csv` - Expand action_label column values
- `annotations/qwen_annotations_original_videos/*.json` - Update conversation targets
- `parkinson_proj/training/convert_to_llava_json.py` - Handle new action labels

**Training Scripts:**
- `parkinson_proj/training/7b/finetune_lora_video.sh` - No changes needed
- `parkinson_proj/training/72b/finetune_lora_video_sbatch.sh` - No changes needed

**Evaluation Scripts:**
- `parkinson_proj/training/evaluation/fine_tuned/eval_lora_video_corrected.py:214-219` - Update confusion matrix classes
- `parkinson_proj/training/evaluation/fine_tuned/convert_to_pd_insighter_format()` - Handle new action types

**Web Application:**
- `parkinson_proj/web_application/core/model_manager.py` - Update action mapping
- `parkinson_proj/web_application/prompts/general_prompts.py` - Expand action descriptions

#### 📋 Implementation Steps:
1. **Data Collection**: Annotate videos with new action labels
2. **Data Balance**: Address class imbalance (current 54% sitting bias)
3. **Prompt Engineering**: Design specific prompts for new actions
4. **Model Retraining**: Fine-tune on expanded action set
5. **Evaluation Metrics**: Develop action-specific performance measures

### 3. ⚡ Web Application Performance Optimization

**Current Limitation**: Slow web app due to repeated processing for same videos.

**Research Opportunity**: Implement caching and preprocessing to accelerate analysis.

#### 🐌 Performance Bottlenecks Identified:

**Model Loading:**
- `parkinson_proj/web_application/core/model_manager.py:46-80` - Model loaded on every request
- Cold start time: ~30-60 seconds for 7B/72B models

**Video Processing:**
- `parkinson_proj/web_application/core/video_annotator.py` - Full video reprocessing
- Frame extraction repeated for identical videos
- No clip-level result caching

**Analysis Pipeline:**
- `parkinson_proj/web_application/core/confidence_analyzer.py` - Redundant confidence calculations
- `parkinson_proj/web_application/core/context_analyzer.py` - Temporal analysis recalculation

#### 🚀 Optimization Strategies:

**1. Intelligent Caching System:**
```
cache/
├── video_hashes/          # Video content hashes
├── clip_features/         # Pre-extracted visual features  
├── model_predictions/     # Cached classification results
├── analysis_results/      # Processed analysis outputs
└── session_data/         # User session persistence
```

**2. Files to Modify:**
- `parkinson_proj/web_application/core/model_manager.py` - Add model persistence
- `parkinson_proj/web_application/utils/video_utils.py` - Add video hashing & caching
- `parkinson_proj/web_application/core/session_manager.py` - Implement result caching
- `parkinson_proj/web_application/web_interface/app.py` - Add cache management UI

**3. Performance Improvements:**
- **Video Hash Checking**: Skip processing for previously analyzed videos
- **Clip-Level Caching**: Store individual clip predictions
- **Feature Precomputation**: Pre-extract and cache visual features
- **Incremental Analysis**: Process only new/changed video segments
- **Model Warm-Up**: Keep model loaded in memory between sessions

**4. Implementation Roadmap:**
1. **Add content-based video hashing** for duplicate detection
2. **Implement clip-level result caching** with SQLite/Redis
3. **Create background processing queue** for large videos
4. **Add progress indicators** for user experience
5. **Optimize model loading** with persistent memory allocation

#### 📊 Expected Performance Gains:
- **First-time analysis**: Current speed
- **Repeated analysis**: 10-50x faster (cache hits)
- **Partial video updates**: 5-10x faster (incremental processing)
- **Session persistence**: Instant loading for recent analyses

## 📚 Implementation Guidelines

### For Each Extension:
1. **Start with data format changes** - Ensure backward compatibility
2. **Implement incremental updates** - Test each component separately  
3. **Maintain evaluation consistency** - Compare against current baselines
4. **Document performance impacts** - Measure speed/accuracy trade-offs
5. **Preserve existing functionality** - Don't break current workflows

### Best Practices:
- **Version control branches** for each major extension
- **Comprehensive testing** on existing video datasets
- **Performance benchmarking** before/after changes
- **User feedback collection** for web app improvements

---

*This document serves as a roadmap for future researchers. Update it as new opportunities and challenges emerge.*