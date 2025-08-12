"""
Context analysis module for generating detailed descriptions of video segments.
"""

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - model-based analysis will be skipped")

from typing import List, Dict, Any, Optional

# Import from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

if TORCH_AVAILABLE:
    from qwen_vl_utils import process_vision_info

from parkinson_proj.web_application.utils.data_structures import AnnotatedClip, VideoSegment
from parkinson_proj.web_application.prompts.context_prompts import (
    GENERAL_DESCRIPTION_PROMPT,
    DETAILED_DESCRIPTION_PROMPT,
    MEDICAL_CONTEXT_PROMPT,
    OBJECT_INTERACTION_PROMPT,
    MOVEMENT_QUALITY_PROMPT
)
from parkinson_proj.web_application.config.analysis_config import load_config
from parkinson_proj.web_application.core.model_manager import model_manager


class ContextAnalyzer:
    """Analyzes video segments and generates detailed descriptions."""
    
    def __init__(self, config_path: str = None):
        """Initialize the context analyzer."""
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
    
    def create_segments_from_clips(self, clips: List[AnnotatedClip]) -> List[VideoSegment]:
        """Create video segments from consecutive clips with the same action."""
        if not clips:
            return []
        
        segments = []
        current_segment = {
            'start_clip': 0,
            'action_type': clips[0].predicted_action,
            'clips': [clips[0]]
        }
        
        for i in range(1, len(clips)):
            if clips[i].predicted_action == current_segment['action_type']:
                # Continue current segment
                current_segment['clips'].append(clips[i])
            else:
                # End current segment
                current_segment['end_clip'] = i - 1
                current_segment['clip_count'] = len(current_segment['clips'])
                
                # Calculate average confidence for this segment
                avg_confidence = sum(clip.confidence_score for clip in current_segment['clips']) / len(current_segment['clips']) if current_segment['clips'] else 0.0
                
                # Create VideoSegment object
                segment = VideoSegment(
                    start_clip=current_segment['start_clip'],
                    end_clip=current_segment['end_clip'],
                    action_type=current_segment['action_type'],
                    clip_count=current_segment['clip_count'],
                    confidence_level=avg_confidence
                )
                segments.append(segment)
                
                # Start new segment
                current_segment = {
                    'start_clip': i,
                    'action_type': clips[i].predicted_action,
                    'clips': [clips[i]]
                }
        
        # Add last segment
        current_segment['end_clip'] = len(clips) - 1
        current_segment['clip_count'] = len(current_segment['clips'])
        
        # Calculate average confidence for last segment
        avg_confidence = sum(clip.confidence_score for clip in current_segment['clips']) / len(current_segment['clips']) if current_segment['clips'] else 0.0
        
        segment = VideoSegment(
            start_clip=current_segment['start_clip'],
            end_clip=current_segment['end_clip'],
            action_type=current_segment['action_type'],
            clip_count=current_segment['clip_count'],
            confidence_level=avg_confidence
        )
        segments.append(segment)
        
        return segments
    
    def generate_segment_description(self, segment: VideoSegment, clips: List[AnnotatedClip], 
                                   description_level: str = "general", medical_context: bool = True) -> str:
        """Generate description for a video segment."""
        if not TORCH_AVAILABLE:
            return "Model not available for description generation."
            
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        if not model or not processor:
            if not self.load_model():
                return "Model not available for description generation."
        
        # Get clips for this segment
        segment_clips = clips[segment.start_clip:segment.end_clip + 1]
        if not segment_clips:
            return "No clips available for this segment."
        
        # Create segment info
        segment_info = self._create_segment_info(segment, segment_clips)
        duration = segment.clip_count * self.config['video']['clip_duration']
        
        # Choose prompt based on description level
        if description_level == "detailed":
            prompt_template = DETAILED_DESCRIPTION_PROMPT
        else:
            prompt_template = GENERAL_DESCRIPTION_PROMPT
        
        # Format the prompt
        prompt = prompt_template.format(
            segment_info=segment_info,
            action_type=segment.action_type,
            duration=duration
        )
        
        # Add medical context if requested
        if medical_context and self.config['context']['medical_context']:
            medical_prompt = MEDICAL_CONTEXT_PROMPT.format(
                segment_info=segment_info,
                action_type=segment.action_type,
                duration=duration
            )
            prompt += "\n\n" + medical_prompt
        
        # Generate description using the model
        description = self._generate_description_with_model(prompt, segment_clips)
        
        return description
    
    def _create_segment_info(self, segment: VideoSegment, clips: List[AnnotatedClip]) -> str:
        """Create information string about the segment."""
        start_time = clips[0].timestamp
        end_time = clips[-1].timestamp + self.config['video']['clip_duration']
        
        info = f"Segment from {start_time:.1f}s to {end_time:.1f}s "
        info += f"({segment.clip_count} clips, {segment.action_type})"
        
        return info
    
    def _generate_description_with_model(self, prompt: str, clips: List[AnnotatedClip]) -> str:
        """Generate description using the model."""
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        
        if not TORCH_AVAILABLE or not model or not processor:
            return "Model not available for description generation."
            
        try:
            # For now, use text-only analysis since we don't have video input in this context
            # In a real implementation, you would pass video clips to the model
            
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
            
            # Generate response with much longer token limit for detailed medical analysis
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024,  # Much longer for detailed medical analysis
                    do_sample=False,
                    temperature=0.1
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            print(f"⚠️  Description generation failed: {e}")
            return f"Description generation failed: {str(e)}"
    
    def analyze_segment_with_video(self, segment: VideoSegment, clips: List[AnnotatedClip], 
                                  description_level: str = "general") -> Dict[str, str]:
        """Analyze a segment with actual video content."""
        if not TORCH_AVAILABLE:
            return {"error": "Model not available"}
            
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        if not model or not processor:
            if not self.load_model():
                return {"error": "Model not available"}
        
        # Get clips for this segment
        segment_clips = clips[segment.start_clip:segment.end_clip + 1]
        if not segment_clips:
            return {"error": "No clips available for this segment"}
        
        # Use the first clip as representative for the segment
        # In a more sophisticated implementation, you might analyze multiple clips
        representative_clip = segment_clips[0]
        
        results = {}
        
        # Generate general description
        if description_level == "general":
            results['general_description'] = self._generate_video_description(
                representative_clip.video_path, 
                GENERAL_DESCRIPTION_PROMPT.format(
                    segment_info=self._create_segment_info(segment, segment_clips),
                    action_type=segment.action_type,
                    duration=segment.clip_count * self.config['video']['clip_duration']
                )
            )
        
        # Generate detailed description
        if description_level == "detailed":
            results['detailed_description'] = self._generate_video_description(
                representative_clip.video_path,
                DETAILED_DESCRIPTION_PROMPT.format(
                    segment_info=self._create_segment_info(segment, segment_clips),
                    action_type=segment.action_type,
                    duration=segment.clip_count * self.config['video']['clip_duration']
                )
            )
            
            # Generate additional analyses
            if self.config['context']['medical_context']:
                results['medical_analysis'] = self._generate_video_description(
                    representative_clip.video_path,
                    MEDICAL_CONTEXT_PROMPT.format(
                        segment_info=self._create_segment_info(segment, segment_clips),
                        action_type=segment.action_type,
                        duration=segment.clip_count * self.config['video']['clip_duration']
                    )
                )
                
                results['movement_quality'] = self._generate_video_description(
                    representative_clip.video_path,
                    MOVEMENT_QUALITY_PROMPT.format(
                        segment_info=self._create_segment_info(segment, segment_clips),
                        action_type=segment.action_type
                    )
                )
        
        return results
    
    def _generate_video_description(self, video_path: str, prompt: str) -> str:
        """Generate description for a video clip."""
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        
        if not TORCH_AVAILABLE or not model or not processor:
            return "Model not available for video description generation."
            
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "min_pixels": 4 * 28 * 28,
                            "max_pixels": 256 * 28 * 28,
                            "total_pixels": 20480 * 28 * 28,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )
            
            # Move inputs to the correct device
            device = self.get_model_device()
            inputs = inputs.to(device)
            
            # Generate response with much longer token limit for detailed medical analysis  
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024,  # Much longer for detailed medical analysis
                    do_sample=False,
                    temperature=0.1
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            print(f"⚠️  Video description generation failed: {e}")
            return f"Description generation failed: {str(e)}"
    
    def analyze_all_segments(self, clips: List[AnnotatedClip], description_level: str = None) -> List[VideoSegment]:
        """Analyze all segments in the video."""
        if description_level is None:
            description_level = self.config['context']['description_level']
        
        # Create segments
        segments = self.create_segments_from_clips(clips)
        
        print(f"📊 Analyzing {len(segments)} segments...")
        
        # Analyze each segment
        for i, segment in enumerate(segments):
            print(f"   Analyzing segment {i+1}/{len(segments)}: {segment.action_type} ({segment.clip_count} clips)")
            
            # Generate both general and detailed descriptions for better analysis
            segment.general_description = self.generate_segment_description(
                segment, clips, "general"
            )
            segment.detailed_description = self.generate_segment_description(
                segment, clips, "detailed"
            )
            
            # Mark as suspicious if confidence is low
            segment_clips = clips[segment.start_clip:segment.end_clip + 1]
            avg_confidence = sum(clip.confidence_score for clip in segment_clips) / len(segment_clips)
            segment.confidence_level = avg_confidence
            
            if avg_confidence < self.config['confidence']['confidence_threshold']:
                segment.is_suspicious = True
        
        print("✅ Segment analysis complete!")
        return segments
    
    def get_segments_by_action(self, segments: List[VideoSegment], action: str) -> List[VideoSegment]:
        """Get all segments with a specific action."""
        return [seg for seg in segments if seg.action_type == action]
    
    def get_suspicious_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Get all segments marked as suspicious."""
        return [seg for seg in segments if seg.is_suspicious]
    
    def generate_segment_report(self, segments: List[VideoSegment]) -> Dict[str, Any]:
        """Generate a comprehensive segment report."""
        if not segments:
            return {}
        
        # Calculate statistics
        total_segments = len(segments)
        suspicious_segments = len([seg for seg in segments if seg.is_suspicious])
        
        # Action-wise statistics
        action_stats = {}
        for segment in segments:
            action = segment.action_type
            if action not in action_stats:
                action_stats[action] = {
                    'count': 0,
                    'total_clips': 0,
                    'avg_confidence': 0.0,
                    'suspicious_count': 0
                }
            
            action_stats[action]['count'] += 1
            action_stats[action]['total_clips'] += segment.clip_count
            action_stats[action]['avg_confidence'] += segment.confidence_level
            if segment.is_suspicious:
                action_stats[action]['suspicious_count'] += 1
        
        # Calculate averages
        for action in action_stats:
            count = action_stats[action]['count']
            action_stats[action]['avg_confidence'] /= count
        
        return {
            'total_segments': total_segments,
            'suspicious_segments': suspicious_segments,
            'action_stats': action_stats,
            'segments': [seg.to_dict() for seg in segments]
        } 