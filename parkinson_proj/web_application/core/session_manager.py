"""
Session manager for coordinating video analysis sessions.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from parkinson_proj.web_application.utils.data_structures import AnnotatedClip, VideoSegment, AnalysisSession
from parkinson_proj.web_application.core.video_annotator import VideoAnnotator
from parkinson_proj.web_application.core.confidence_analyzer import ConfidenceAnalyzer
from parkinson_proj.web_application.core.context_analyzer import ContextAnalyzer
from parkinson_proj.web_application.config.analysis_config import load_config
from parkinson_proj.web_application.core.model_manager import model_manager


class SessionManager:
    """Manages analysis sessions for interactive video analysis."""
    
    def __init__(self, config_path: str = None):
        """Initialize the session manager."""
        self.config = load_config(config_path)
        self.video_annotator = VideoAnnotator(config_path)
        self.confidence_analyzer = ConfidenceAnalyzer(config_path)
        self.context_analyzer = ContextAnalyzer(config_path)
        self.current_session = None
        
        # Initialize model at startup
        print("🚀 Initializing model manager...")
        model_manager.load_model()
        
    def create_session(self, video_folder: str, max_clips: Optional[int] = None) -> AnalysisSession:
        """Create a new analysis session for a video folder."""
        print(f"🚀 Creating analysis session for {video_folder}")
        
        # Check if video folder exists
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")
        
        # Annotate video clips
        print("📹 Annotating video clips...")
        annotated_clips = self.video_annotator.annotate_video_folder(video_folder, max_clips)
        
        if not annotated_clips:
            raise ValueError("No video clips were annotated")
        
        # Analyze confidence
        print("🔍 Analyzing confidence...")
        annotated_clips = self.confidence_analyzer.analyze_confidence(annotated_clips)
        
        # Create segments
        print("📊 Creating video segments...")
        segments = self.context_analyzer.create_segments_from_clips(annotated_clips)
        
        # Create session
        self.current_session = AnalysisSession(
            video_folder=video_folder,
            annotated_clips=annotated_clips,
            segments=segments
        )
        
        print(f"✅ Session created successfully!")
        print(f"   Clips: {len(annotated_clips)}")
        print(f"   Segments: {len(segments)}")
        
        return self.current_session
    
    def load_session(self, session_file: str) -> AnalysisSession:
        """Load an existing analysis session from file."""
        print(f"📂 Loading session from {session_file}")
        
        if not os.path.exists(session_file):
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct clips
        annotated_clips = [AnnotatedClip.from_dict(clip_data) for clip_data in session_data['clips']]
        
        # Reconstruct segments
        segments = [VideoSegment.from_dict(segment_data) for segment_data in session_data['segments']]
        
        # Create session
        self.current_session = AnalysisSession(
            video_folder=session_data['video_folder'],
            annotated_clips=annotated_clips,
            segments=segments,
            confidence_level=session_data.get('confidence_level', 2),
            description_level=session_data.get('description_level', 'general'),
            medical_context=session_data.get('medical_context', True)
        )
        
        print(f"✅ Session loaded successfully!")
        print(f"   Clips: {len(annotated_clips)}")
        print(f"   Segments: {len(segments)}")
        
        return self.current_session
    
    def save_session(self, session: AnalysisSession, output_file: str):
        """Save an analysis session to file."""
        print(f"💾 Saving session to {output_file}")
        
        session_data = {
            'video_folder': session.video_folder,
            'confidence_level': session.confidence_level,
            'description_level': session.description_level,
            'medical_context': session.medical_context,
            'clips': [clip.to_dict() for clip in session.annotated_clips],
            'segments': [segment.to_dict() for segment in session.segments]
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print("✅ Session saved successfully!")
    
    def get_suspicious_clips(self, session: AnalysisSession) -> List[AnnotatedClip]:
        """Get all suspicious clips from the session."""
        return session.get_suspicious_clips()
    
    def get_segments_by_action(self, session: AnalysisSession, action: str) -> List[VideoSegment]:
        """Get all segments with a specific action."""
        return session.get_segments_by_action(action)
    
    def generate_detailed_description(self, session: AnalysisSession, segment: VideoSegment) -> Dict[str, str]:
        """Generate detailed description for a specific segment."""
        # Model is managed by model_manager, no need to check here
        return self.context_analyzer.analyze_segment_with_video(
            segment, 
            session.annotated_clips, 
            "detailed"
        )
    
    def update_confidence_analysis(self, session: AnalysisSession, strictness_level: int = None) -> AnalysisSession:
        """Update confidence analysis with different strictness level."""
        print(f"🔄 Updating confidence analysis...")
        
        if strictness_level is None:
            strictness_level = session.confidence_level
        
        # Re-analyze confidence
        updated_clips = self.confidence_analyzer.analyze_confidence(
            session.annotated_clips, 
            strictness_level
        )
        
        # Update session
        session.annotated_clips = updated_clips
        session.confidence_level = strictness_level
        
        # Update segments
        session.segments = self.context_analyzer.create_segments_from_clips(updated_clips)
        
        print("✅ Confidence analysis updated!")
        return session
    
    def update_context_analysis(self, session: AnalysisSession, description_level: str = None) -> AnalysisSession:
        """Update context analysis with different description level."""
        print(f"🔄 Updating context analysis...")
        
        if description_level is None:
            description_level = session.description_level
        
        # Re-analyze segments
        updated_segments = self.context_analyzer.analyze_all_segments(
            session.annotated_clips, 
            description_level
        )
        
        # Update session
        session.segments = updated_segments
        session.description_level = description_level
        
        print("✅ Context analysis updated!")
        return session
    
    def generate_comprehensive_report(self, session: AnalysisSession) -> Dict[str, Any]:
        """Generate a comprehensive report for the session."""
        print("📊 Generating comprehensive report...")
        
        # Confidence report
        confidence_report = self.confidence_analyzer.generate_confidence_report(session.annotated_clips)
        
        # Segment report
        segment_report = self.context_analyzer.generate_segment_report(session.segments)
        
        # Combine reports
        comprehensive_report = {
            'session_info': {
                'video_folder': session.video_folder,
                'total_clips': len(session.annotated_clips),
                'total_segments': len(session.segments),
                'confidence_level': session.confidence_level,
                'description_level': session.description_level,
                'medical_context': session.medical_context
            },
            'confidence_analysis': confidence_report,
            'segment_analysis': segment_report,
            'summary': {
                'suspicious_clips_count': len(session.get_suspicious_clips()),
                'suspicious_segments_count': len([seg for seg in session.segments if seg.is_suspicious]),
                'action_distribution': self._get_action_distribution(session),
                'confidence_distribution': self._get_confidence_distribution(session)
            }
        }
        
        print("✅ Comprehensive report generated!")
        return comprehensive_report
    
    def _get_action_distribution(self, session: AnalysisSession) -> Dict[str, int]:
        """Get distribution of actions in the session."""
        action_counts = {}
        for clip in session.annotated_clips:
            action = clip.predicted_action
            action_counts[action] = action_counts.get(action, 0) + 1
        return action_counts
    
    def _get_confidence_distribution(self, session: AnalysisSession) -> Dict[str, int]:
        """Get distribution of confidence levels in the session."""
        confidence_ranges = {
            'high': 0,      # 0.8-1.0
            'medium': 0,    # 0.5-0.8
            'low': 0,       # 0.2-0.5
            'very_low': 0   # 0.0-0.2
        }
        
        for clip in session.annotated_clips:
            score = clip.confidence_score
            if score >= 0.8:
                confidence_ranges['high'] += 1
            elif score >= 0.5:
                confidence_ranges['medium'] += 1
            elif score >= 0.2:
                confidence_ranges['low'] += 1
            else:
                confidence_ranges['very_low'] += 1
        
        return confidence_ranges
    
    def query_session(self, session: AnalysisSession, query: str) -> Dict[str, Any]:
        """Process a natural language query against the session."""
        query = query.lower().strip()
        
        if "suspicious" in query or "wrong" in query or "confidence" in query:
            # Confidence-related query
            suspicious_clips = self.get_suspicious_clips(session)
            return {
                'query_type': 'confidence',
                'suspicious_clips': [clip.to_dict() for clip in suspicious_clips],
                'count': len(suspicious_clips),
                'message': f"Found {len(suspicious_clips)} suspicious clips"
            }
        
        elif "walking" in query:
            # Action-specific query
            segments = self.get_segments_by_action(session, "walking")
            return {
                'query_type': 'action',
                'action': 'walking',
                'segments': [seg.to_dict() for seg in segments],
                'count': len(segments),
                'message': f"Found {len(segments)} walking segments"
            }
        
        elif "sitting" in query:
            segments = self.get_segments_by_action(session, "sitting")
            return {
                'query_type': 'action',
                'action': 'sitting',
                'segments': [seg.to_dict() for seg in segments],
                'count': len(segments),
                'message': f"Found {len(segments)} sitting segments"
            }
        
        elif "standing" in query:
            segments = self.get_segments_by_action(session, "standing")
            return {
                'query_type': 'action',
                'action': 'standing',
                'segments': [seg.to_dict() for seg in segments],
                'count': len(segments),
                'message': f"Found {len(segments)} standing segments"
            }
        
        elif "description" in query or "detail" in query:
            # Description-related query
            return {
                'query_type': 'description',
                'message': "Use generate_detailed_description() for specific segments",
                'available_segments': [seg.to_dict() for seg in session.segments[:5]]  # First 5 segments
            }
        
        else:
            # General query - return summary
            return {
                'query_type': 'general',
                'message': "Session summary",
                'total_clips': len(session.annotated_clips),
                'total_segments': len(session.segments),
                'suspicious_clips': len(session.get_suspicious_clips()),
                'action_distribution': self._get_action_distribution(session)
            } 