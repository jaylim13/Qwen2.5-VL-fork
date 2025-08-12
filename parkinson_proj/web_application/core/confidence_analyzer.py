"""
Confidence analysis module for detecting potentially wrong predictions.
"""

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - model-based analysis will be skipped")

from typing import List, Dict, Any, Tuple

# Import from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

if TORCH_AVAILABLE:
    from qwen_vl_utils import process_vision_info

from parkinson_proj.web_application.utils.data_structures import AnnotatedClip
from parkinson_proj.web_application.utils.temporal_consistency import (
    calculate_temporal_consistency, 
    detect_anomalous_transitions,
    calculate_confidence_scores
)
from parkinson_proj.web_application.prompts.confidence_prompts import CONFIDENCE_PROMPTS
from parkinson_proj.web_application.config.analysis_config import load_config
from parkinson_proj.web_application.core.model_manager import model_manager


class ConfidenceAnalyzer:
    """Analyzes confidence of video action predictions."""
    
    def __init__(self, config_path: str = None):
        """Initialize the confidence analyzer."""
        self.config = load_config(config_path)
        # Model will be loaded by model_manager when needed
        
    def load_model(self):
        """Load the Qwen2.5-VL model using model manager."""
        return model_manager.load_model()
    
    def get_model_device(self):
        """Get the main device of the model."""
        model = model_manager.get_model()
        if not model:
            return None
        if hasattr(model, 'hf_device_map'):
            return list(model.hf_device_map.values())[0]
        else:
            return next(model.parameters()).device
    
    def analyze_confidence(self, clips: List[AnnotatedClip], strictness_level: int = None) -> List[AnnotatedClip]:
        """Analyze confidence of all clips and mark suspicious ones."""
        if not clips:
            return []
        
        if strictness_level is None:
            strictness_level = self.config['confidence']['strictness_level']
        
        print(f"🔍 Analyzing confidence with strictness level {strictness_level}...")
        
        # 1. Calculate temporal consistency scores
        temporal_scores = calculate_temporal_consistency(
            clips, 
            self.config['confidence']['temporal_window_size']
        )
        
        # 2. Calculate comprehensive confidence scores
        confidence_scores = calculate_confidence_scores(clips, self.config)
        
        # 3. Detect anomalous transitions
        suspicious_indices = detect_anomalous_transitions(clips, strictness_level)
        
        # 4. Update clips with confidence information
        updated_clips = []
        for i, clip in enumerate(clips):
            # Update confidence scores
            clip.temporal_consistency = temporal_scores[i]
            clip.confidence_score = confidence_scores[i]
            
            # Mark as suspicious if in suspicious indices
            clip.is_suspicious = i in suspicious_indices
            
            updated_clips.append(clip)
        
        # 5. Use model-based analysis for high-confidence suspicious clips (if available)
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        if TORCH_AVAILABLE and model and processor:
            updated_clips = self._model_based_confidence_analysis(updated_clips, strictness_level)
        
        print(f"✅ Confidence analysis complete! Found {len(suspicious_indices)} suspicious clips.")
        return updated_clips
    
    def _model_based_confidence_analysis(self, clips: List[AnnotatedClip], strictness_level: int) -> List[AnnotatedClip]:
        """Use the model to analyze suspicious clips more deeply."""
        if not clips or not TORCH_AVAILABLE:
            return clips
        
        # Get the appropriate prompt for the strictness level
        prompt_template = CONFIDENCE_PROMPTS.get(strictness_level, CONFIDENCE_PROMPTS[2])
        
        # Group clips into sequences for analysis
        sequences = self._create_analysis_sequences(clips)
        
        for sequence_info in sequences:
            start_idx, end_idx, sequence_clips = sequence_info
            
            # Create sequence information for the prompt
            clip_sequence = [f"Clip {i}: {clip.clip_id}" for i, clip in enumerate(sequence_clips)]
            action_sequence = [clip.predicted_action for clip in sequence_clips]
            
            # Format the prompt
            prompt = prompt_template.format(
                clip_sequence=", ".join(clip_sequence),
                action_sequence=", ".join(action_sequence)
            )
            
            # Analyze with model
            analysis_result = self._analyze_sequence_with_model(prompt, sequence_clips)
            
            # Update suspicious flags based on model analysis
            for i, clip in enumerate(sequence_clips):
                global_idx = start_idx + i
                if global_idx < len(clips):
                    # Update based on model analysis
                    if analysis_result.get(f"clip_{i}", False):
                        clips[global_idx].is_suspicious = True
                        clips[global_idx].confidence_score *= 0.5  # Reduce confidence
        
        return clips
    
    def _create_analysis_sequences(self, clips: List[AnnotatedClip], sequence_length: int = 10) -> List[Tuple[int, int, List[AnnotatedClip]]]:
        """Create sequences of clips for analysis."""
        sequences = []
        
        for i in range(0, len(clips), sequence_length):
            end_idx = min(i + sequence_length, len(clips))
            sequence_clips = clips[i:end_idx]
            sequences.append((i, end_idx, sequence_clips))
        
        return sequences
    
    def _analyze_sequence_with_model(self, prompt: str, clips: List[AnnotatedClip]) -> Dict[str, bool]:
        """Analyze a sequence of clips using the model."""
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        
        if not TORCH_AVAILABLE or not model or not processor:
            return {}
            
        try:
            # Create messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to the correct device
            device = self.get_model_device()
            inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=128,  # Longer response for analysis
                    do_sample=False,
                    temperature=0.1
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Parse the response to extract suspicious clip indices
            analysis_result = self._parse_confidence_response(output_text[0], len(clips))
            return analysis_result
            
        except Exception as e:
            print(f"⚠️  Model-based confidence analysis failed: {e}")
            return {}
    
    def _parse_confidence_response(self, response: str, num_clips: int) -> Dict[str, bool]:
        """Parse the model's confidence analysis response."""
        result = {}
        
        # Look for patterns like "Clip X:" in the response
        import re
        clip_pattern = r"Clip\s+(\d+):"
        matches = re.findall(clip_pattern, response)
        
        for match in matches:
            try:
                clip_idx = int(match)
                if 0 <= clip_idx < num_clips:
                    result[f"clip_{clip_idx}"] = True
            except ValueError:
                continue
        
        return result
    
    def get_suspicious_clips(self, clips: List[AnnotatedClip]) -> List[AnnotatedClip]:
        """Get all clips marked as suspicious."""
        return [clip for clip in clips if clip.is_suspicious]
    
    def get_low_confidence_clips(self, clips: List[AnnotatedClip], threshold: float = None) -> List[AnnotatedClip]:
        """Get clips with confidence below threshold."""
        if threshold is None:
            threshold = self.config['confidence']['confidence_threshold']
        
        return [clip for clip in clips if clip.confidence_score < threshold]
    
    def generate_confidence_report(self, clips: List[AnnotatedClip]) -> Dict[str, Any]:
        """Generate a comprehensive confidence report."""
        if not clips:
            return {}
        
        suspicious_clips = self.get_suspicious_clips(clips)
        low_confidence_clips = self.get_low_confidence_clips(clips)
        
        # Calculate statistics
        total_clips = len(clips)
        suspicious_count = len(suspicious_clips)
        low_confidence_count = len(low_confidence_clips)
        
        # Average confidence scores
        avg_confidence = sum(clip.confidence_score for clip in clips) / total_clips
        avg_temporal_consistency = sum(clip.temporal_consistency for clip in clips) / total_clips
        
        # Action-wise statistics
        action_stats = {}
        for clip in clips:
            action = clip.predicted_action
            if action not in action_stats:
                action_stats[action] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'suspicious_count': 0
                }
            
            action_stats[action]['count'] += 1
            action_stats[action]['avg_confidence'] += clip.confidence_score
            if clip.is_suspicious:
                action_stats[action]['suspicious_count'] += 1
        
        # Calculate averages
        for action in action_stats:
            count = action_stats[action]['count']
            action_stats[action]['avg_confidence'] /= count
        
        return {
            'total_clips': total_clips,
            'suspicious_clips': suspicious_count,
            'low_confidence_clips': low_confidence_count,
            'avg_confidence': avg_confidence,
            'avg_temporal_consistency': avg_temporal_consistency,
            'action_stats': action_stats,
            'suspicious_clip_ids': [clip.clip_id for clip in suspicious_clips],
            'low_confidence_clip_ids': [clip.clip_id for clip in low_confidence_clips]
        } 