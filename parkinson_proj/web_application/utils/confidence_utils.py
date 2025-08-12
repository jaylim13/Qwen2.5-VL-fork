"""
Confidence calculation utilities ported from the original system.
"""

from typing import List, Dict, Any
from collections import defaultdict
from utils.data_structures import AnnotatedClip


def calculate_temporal_consistency(clips: List[AnnotatedClip], window_size: int = 5) -> List[float]:
    """Calculate temporal consistency scores for each clip."""
    if len(clips) == 0:
        return []
    
    consistency_scores = []
    
    for i, clip in enumerate(clips):
        # Get surrounding clips within window
        start_idx = max(0, i - window_size)
        end_idx = min(len(clips), i + window_size + 1)
        
        surrounding_clips = clips[start_idx:end_idx]
        surrounding_actions = [c.predicted_action for c in surrounding_clips if c != clip]
        
        if not surrounding_actions:
            consistency_scores.append(0.5)  # Neutral if no context
            continue
        
        # Calculate consistency ratio
        same_action_count = surrounding_actions.count(clip.predicted_action)
        total_surrounding = len(surrounding_actions)
        consistency_ratio = same_action_count / total_surrounding
        
        consistency_scores.append(consistency_ratio)
    
    return consistency_scores


def calculate_confidence_scores(clips: List[AnnotatedClip], config: Dict[str, Any]) -> List[float]:
    """Calculate comprehensive confidence scores for clips."""
    if not clips:
        return []
    
    # Get configuration parameters
    temporal_weight = 0.4
    response_quality_weight = 0.3
    model_uncertainty_weight = 0.3
    
    # Default window size if not in config
    window_size = 5
    if config and 'confidence' in config and 'temporal_window_size' in config['confidence']:
        window_size = config['confidence']['temporal_window_size']
    
    # 1. Temporal consistency scores
    temporal_scores = calculate_temporal_consistency(clips, window_size)
    
    # 2. Response quality scores
    response_scores = []
    for clip in clips:
        score = 0.0
        
        # Exact match with predefined actions
        if clip.predicted_action in ['walking', 'sitting', 'standing', 'go upstair', 'go downstair']:
            score += 0.3
        
        # Response quality
        if clip.raw_response == clip.predicted_action:
            score += 0.2  # Exact match
        elif clip.predicted_action != "unknown":
            score += 0.1  # Partial match
        
        # Response length (reasonable responses)
        response_length = len(clip.raw_response) if clip.raw_response else 0
        if 5 <= response_length <= 50:
            score += 0.1
        
        # Error handling
        if clip.error_message:
            score = 0.0
        
        response_scores.append(min(score, 1.0))
    
    # 3. Model uncertainty scores (simplified)
    uncertainty_scores = []
    for clip in clips:
        if clip.error_message:
            score = 0.0
        elif clip.predicted_action == "unknown":
            score = 0.3
        elif clip.predicted_action == "error":
            score = 0.0
        else:
            score = 0.8  # Assume good if no errors
        uncertainty_scores.append(score)
    
    # Combine scores
    final_scores = []
    for i in range(len(clips)):
        final_score = (
            temporal_weight * temporal_scores[i] +
            response_quality_weight * response_scores[i] +
            model_uncertainty_weight * uncertainty_scores[i]
        )
        final_scores.append(final_score)
    
    return final_scores


def update_clip_confidence_scores(clips: List[AnnotatedClip], config: Dict[str, Any]) -> List[AnnotatedClip]:
    """Update confidence scores for all clips in place."""
    if not clips:
        return clips
    
    # Calculate confidence scores
    confidence_scores = calculate_confidence_scores(clips, config)
    
    # Update clips with calculated scores
    for i, clip in enumerate(clips):
        if i < len(confidence_scores):
            clip.confidence_score = confidence_scores[i]
        else:
            clip.confidence_score = 0.0
    
    return clips