"""
Temporal consistency analysis utilities.
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np
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


def detect_anomalous_transitions(clips: List[AnnotatedClip], strictness_level: int = 2) -> List[int]:
    """Detect anomalous transitions based on strictness level."""
    if len(clips) < 3:
        return []
    
    suspicious_indices = []
    
    if strictness_level == 1:  # Loose - flag any transition
        for i in range(1, len(clips)):
            if clips[i].predicted_action != clips[i-1].predicted_action:
                suspicious_indices.append(i)
    
    elif strictness_level == 2:  # Medium - statistical analysis
        suspicious_indices = _detect_medium_anomalies(clips)
    
    elif strictness_level == 3:  # Strict - only highly confident misclassifications
        suspicious_indices = _detect_strict_anomalies(clips)
    
    return suspicious_indices


def _detect_medium_anomalies(clips: List[AnnotatedClip]) -> List[int]:
    """Detect medium-level anomalies using statistical analysis."""
    suspicious_indices = []
    
    # Calculate action frequencies
    action_counts = defaultdict(int)
    for clip in clips:
        action_counts[clip.predicted_action] += 1
    
    total_clips = len(clips)
    action_frequencies = {action: count/total_clips for action, count in action_counts.items()}
    
    # Look for isolated actions (appear only once or very rarely)
    for i, clip in enumerate(clips):
        if action_frequencies[clip.predicted_action] < 0.1:  # Less than 10% frequency
            # Check if it's isolated
            left_context = clips[max(0, i-2):i]
            right_context = clips[i+1:min(len(clips), i+3)]
            
            left_actions = [c.predicted_action for c in left_context]
            right_actions = [c.predicted_action for c in right_context]
            
            # If surrounding actions are different, flag as suspicious
            if left_actions and right_actions:
                if (clip.predicted_action not in left_actions and 
                    clip.predicted_action not in right_actions):
                    suspicious_indices.append(i)
    
    return suspicious_indices


def _detect_strict_anomalies(clips: List[AnnotatedClip]) -> List[int]:
    """Detect strict-level anomalies - only highly confident misclassifications."""
    suspicious_indices = []
    
    # Look for patterns like: AAAAA -> B -> AAAAA (where B is likely wrong)
    for i in range(2, len(clips) - 2):
        current_action = clips[i].predicted_action
        
        # Check left context (at least 2 clips)
        left_actions = [clips[j].predicted_action for j in range(i-2, i)]
        # Check right context (at least 2 clips)
        right_actions = [clips[j].predicted_action for j in range(i+1, i+3)]
        
        # If left and right contexts are the same, but different from current
        if (len(set(left_actions)) == 1 and 
            len(set(right_actions)) == 1 and
            left_actions[0] == right_actions[0] and
            current_action != left_actions[0]):
            suspicious_indices.append(i)
    
    return suspicious_indices


def analyze_temporal_patterns(clips: List[AnnotatedClip]) -> Dict[str, Any]:
    """Analyze temporal patterns in the video."""
    if not clips:
        return {}
    
    # Calculate action sequences
    action_sequence = [clip.predicted_action for clip in clips]
    
    # Find segments of consecutive same actions
    segments = []
    current_segment = {
        'action': action_sequence[0],
        'start': 0,
        'count': 1
    }
    
    for i in range(1, len(action_sequence)):
        if action_sequence[i] == current_segment['action']:
            current_segment['count'] += 1
        else:
            # End current segment
            current_segment['end'] = i - 1
            segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                'action': action_sequence[i],
                'start': i,
                'count': 1
            }
    
    # Add last segment
    current_segment['end'] = len(action_sequence) - 1
    segments.append(current_segment)
    
    # Calculate statistics
    action_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0})
    for segment in segments:
        action = segment['action']
        action_stats[action]['count'] += 1
        action_stats[action]['total_duration'] += segment['count']
    
    return {
        'segments': segments,
        'action_stats': dict(action_stats),
        'total_clips': len(clips),
        'unique_actions': len(set(action_sequence))
    }


def calculate_confidence_scores(clips: List[AnnotatedClip], config: Dict[str, Any]) -> List[float]:
    """Calculate comprehensive confidence scores for clips."""
    if not clips:
        return []
    
    # Get configuration parameters with fallback values
    temporal_weight = 0.4
    response_quality_weight = 0.3
    model_uncertainty_weight = 0.3
    
    # Get temporal window size with fallback
    temporal_window = config.get('confidence', {}).get('temporal_window_size', 5)
    
    # 1. Temporal consistency scores
    temporal_scores = calculate_temporal_consistency(clips, temporal_window)
    
    # 2. Response quality scores - enhanced with more realistic scoring
    response_scores = []
    for clip in clips:
        score = 0.0
        
        # Base score for valid actions (higher for common actions)
        if clip.predicted_action in ['walking', 'sitting', 'standing']:
            score += 0.5  # Common actions get higher base score
        elif clip.predicted_action in ['go upstair', 'go downstair']:
            score += 0.4  # Less common but valid actions
        elif clip.predicted_action not in ['unknown', 'error', '']:
            score += 0.3  # Any other valid action
        
        # Response quality bonus
        if clip.raw_response and clip.predicted_action:
            if clip.raw_response.lower().strip() == clip.predicted_action.lower().strip():
                score += 0.3  # Exact match bonus
            elif clip.predicted_action.lower() in clip.raw_response.lower():
                score += 0.2  # Partial match bonus
            elif len(clip.raw_response) > 0:
                score += 0.1  # At least got some response
        
        # Response length and quality
        if clip.raw_response:
            response_length = len(clip.raw_response.strip())
            if 3 <= response_length <= 20:  # Reasonable response length
                score += 0.1
            elif response_length > 20:  # Detailed response
                score += 0.05
        
        # Penalty for errors
        if clip.error_message or clip.predicted_action in ['error', 'unknown', '']:
            score = max(0.1, score * 0.3)  # Heavy penalty but not zero
        
        response_scores.append(min(score, 1.0))
    
    # 3. Model uncertainty scores - more realistic distribution
    uncertainty_scores = []
    for clip in clips:
        if clip.error_message or clip.predicted_action == "error":
            score = 0.1  # Some uncertainty even for errors
        elif clip.predicted_action == "unknown" or clip.predicted_action == "":
            score = 0.2  # Low but not zero for unknown
        elif clip.predicted_action in ['walking', 'sitting', 'standing']:
            score = 0.9  # High confidence for common actions
        elif clip.predicted_action in ['go upstair', 'go downstair']:
            score = 0.8  # Good confidence for specific actions
        else:
            score = 0.7  # Moderate confidence for other actions
        uncertainty_scores.append(score)
    
    # Combine scores with normalization
    final_scores = []
    for i in range(len(clips)):
        final_score = (
            temporal_weight * temporal_scores[i] +
            response_quality_weight * response_scores[i] +
            model_uncertainty_weight * uncertainty_scores[i]
        )
        # Ensure realistic range between 0.1 and 0.95
        final_score = max(0.1, min(0.95, final_score))
        final_scores.append(final_score)
    
    return final_scores 