"""
Data structures for interactive video analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class AnnotatedClip:
    """Represents a single annotated video clip."""
    clip_id: str
    video_path: str
    timestamp: float
    predicted_action: str
    raw_response: str
    confidence_score: float = 0.0
    temporal_consistency: float = 0.0
    is_suspicious: bool = False
    general_description: str = ""
    detailed_description: str = ""
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "timestamp": self.timestamp,
            "predicted_action": self.predicted_action,
            "raw_response": self.raw_response,
            "confidence_score": self.confidence_score,
            "temporal_consistency": self.temporal_consistency,
            "is_suspicious": self.is_suspicious,
            "general_description": self.general_description,
            "detailed_description": self.detailed_description,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotatedClip':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VideoSegment:
    """Represents a segment of consecutive clips with the same action."""
    start_clip: int
    end_clip: int
    action_type: str
    clip_count: int
    general_description: str = ""
    detailed_description: str = ""
    confidence_level: float = 0.0
    is_suspicious: bool = False
    segment_video_path: str = ""  # Path to combined segment video
    transition_video_path: str = ""  # Path to transition video (if applicable)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_clip": self.start_clip,
            "end_clip": self.end_clip,
            "action_type": self.action_type,
            "clip_count": self.clip_count,
            "general_description": self.general_description,
            "detailed_description": self.detailed_description,
            "confidence_level": self.confidence_level,
            "is_suspicious": self.is_suspicious,
            "segment_video_path": self.segment_video_path,
            "transition_video_path": self.transition_video_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoSegment':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AnalysisSession:
    """Represents an analysis session for a video."""
    video_folder: str
    annotated_clips: List[AnnotatedClip]
    segments: List[VideoSegment]
    confidence_level: int = 2
    description_level: str = "general"
    medical_context: bool = True
    
    def get_clip_by_id(self, clip_id: str) -> Optional[AnnotatedClip]:
        """Get clip by ID."""
        for clip in self.annotated_clips:
            if clip.clip_id == clip_id:
                return clip
        return None
    
    def get_segments_by_action(self, action: str) -> List[VideoSegment]:
        """Get all segments with a specific action."""
        return [seg for seg in self.segments if seg.action_type == action]
    
    def get_suspicious_clips(self) -> List[AnnotatedClip]:
        """Get all clips marked as suspicious."""
        return [clip for clip in self.annotated_clips if clip.is_suspicious] 