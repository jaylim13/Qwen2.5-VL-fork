#!/usr/bin/env python3
"""
Command-line interface for interactive video analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the parkinson_proj directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from parkinson_proj.web_application.core.session_manager import SessionManager
from parkinson_proj.web_application.utils.data_structures import AnalysisSession
from parkinson_proj.web_application.config.analysis_config import load_config


class InteractiveVideoAnalysisCLI:
    """Command-line interface for interactive video analysis."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.config = load_config()
        print("🔄 Initializing system and loading model...")
        self.session_manager = SessionManager()
        self.current_session = None
        
    def run(self):
        """Run the interactive CLI."""
        print("🎬 Interactive Video Analysis System")
        print("=" * 50)
        print("Medical video analysis with confidence detection and detailed descriptions.")
        print()
        
        while True:
            self._show_menu()
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                self._load_video()
            elif choice == "2":
                self._analyze_video()
            elif choice == "3":
                self._show_session_info()
            elif choice == "4":
                self._query_session()
            elif choice == "5":
                self._show_segments()
            elif choice == "6":
                self._get_detailed_description()
            elif choice == "7":
                self._save_session()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _show_menu(self):
        """Show the main menu."""
        print("\n📋 Main Menu:")
        print("1. Load Video")
        print("2. Analyze Video")
        print("3. Show Session Info")
        print("4. Query Session")
        print("5. Show Segments")
        print("6. Get Detailed Description")
        print("7. Save Session")
        print("8. Exit")
    
    def _load_video(self):
        """Load a video folder."""
        print("\n📁 Load Video")
        print("-" * 30)
        
        video_folder = input("Enter video folder path (default: ../../data/video_0): ").strip()
        if not video_folder:
            video_folder = "../../data/video_0"
        
        max_clips = input("Enter max clips to process (default: 50): ").strip()
        if not max_clips:
            max_clips = 50
        else:
            try:
                max_clips = int(max_clips)
            except ValueError:
                max_clips = 50
        
        try:
            print(f"🔄 Loading video from {video_folder}...")
            session = self.session_manager.create_session(video_folder, max_clips)
            self.current_session = session
            
            print(f"✅ Video loaded successfully!")
            print(f"   Clips: {len(session.annotated_clips)}")
            print(f"   Segments: {len(session.segments)}")
            
        except Exception as e:
            print(f"❌ Failed to load video: {e}")
    
    def _analyze_video(self):
        """Analyze the current video session."""
        if not self.current_session:
            print("❌ No session loaded. Please load a video first.")
            return
        
        print("\n🔍 Analyze Video")
        print("-" * 30)
        
        print("Confidence Level:")
        print("1. Loose (flag any transition)")
        print("2. Medium (statistical analysis)")
        print("3. Strict (highly confident misclassifications)")
        
        conf_choice = input("Enter confidence level (1-3, default: 2): ").strip()
        if not conf_choice:
            conf_level = 2
        else:
            try:
                conf_level = int(conf_choice)
                if conf_level not in [1, 2, 3]:
                    conf_level = 2
            except ValueError:
                conf_level = 2
        
        print("Description Level:")
        print("1. General")
        print("2. Detailed")
        
        desc_choice = input("Enter description level (1-2, default: 1): ").strip()
        if not desc_choice:
            desc_level = "general"
        else:
            desc_level = "detailed" if desc_choice == "2" else "general"
        
        medical_context = input("Enable medical context analysis? (y/n, default: y): ").strip().lower()
        medical_context = medical_context != "n"
        
        try:
            print(f"🔄 Analyzing video with confidence level {conf_level}...")
            
            # Update session settings
            self.current_session.confidence_level = conf_level
            self.current_session.description_level = desc_level
            self.current_session.medical_context = medical_context
            
            # Update confidence analysis
            self.current_session = self.session_manager.update_confidence_analysis(
                self.current_session, conf_level
            )
            
            # Update context analysis
            self.current_session = self.session_manager.update_context_analysis(
                self.current_session, desc_level
            )
            
            # Generate comprehensive report
            report = self.session_manager.generate_comprehensive_report(self.current_session)
            
            print("✅ Analysis complete!")
            print(f"   Suspicious clips: {len(self.current_session.get_suspicious_clips())}")
            print(f"   Action distribution: {report['summary']['action_distribution']}")
            
        except Exception as e:
            print(f"❌ Failed to analyze video: {e}")
    
    def _show_session_info(self):
        """Show current session information."""
        if not self.current_session:
            print("❌ No session loaded.")
            return
        
        print("\n📊 Session Information")
        print("-" * 30)
        print(f"Video folder: {self.current_session.video_folder}")
        print(f"Total clips: {len(self.current_session.annotated_clips)}")
        print(f"Total segments: {len(self.current_session.segments)}")
        print(f"Confidence level: {self.current_session.confidence_level}")
        print(f"Description level: {self.current_session.description_level}")
        print(f"Medical context: {self.current_session.medical_context}")
        
        # Show action distribution
        action_dist = self.session_manager._get_action_distribution(self.current_session)
        print(f"\nAction distribution:")
        for action, count in action_dist.items():
            print(f"  {action}: {count}")
        
        # Show suspicious clips
        suspicious_clips = self.current_session.get_suspicious_clips()
        if suspicious_clips:
            print(f"\nSuspicious clips ({len(suspicious_clips)}):")
            for clip in suspicious_clips[:5]:  # Show first 5
                print(f"  {clip.clip_id}: {clip.predicted_action} (confidence: {clip.confidence_score:.2f})")
            if len(suspicious_clips) > 5:
                print(f"  ... and {len(suspicious_clips) - 5} more")
    
    def _query_session(self):
        """Query the current session."""
        if not self.current_session:
            print("❌ No session loaded. Please load a video first.")
            return
        
        print("\n💬 Query Session")
        print("-" * 30)
        print("Example queries:")
        print("- 'What clips do you think are wrong?'")
        print("- 'Show me walking segments'")
        print("- 'Find sitting segments'")
        print("- 'General summary'")
        
        query = input("\nEnter your query: ").strip()
        if not query:
            return
        
        try:
            result = self.session_manager.query_session(self.current_session, query)
            
            print(f"\n🔍 Query Results:")
            print(f"Type: {result.get('query_type', 'unknown')}")
            print(f"Message: {result.get('message', 'No message')}")
            
            if 'count' in result:
                print(f"Count: {result['count']}")
            
            if 'suspicious_clips' in result and result['suspicious_clips']:
                print(f"\nSuspicious clips:")
                for clip in result['suspicious_clips'][:3]:  # Show first 3
                    print(f"  {clip['clip_id']}: {clip['predicted_action']}")
            
            if 'segments' in result and result['segments']:
                print(f"\nSegments:")
                for segment in result['segments'][:3]:  # Show first 3
                    print(f"  {segment['action_type']}: {segment['clip_count']} clips")
            
        except Exception as e:
            print(f"❌ Failed to process query: {e}")
    
    def _show_segments(self):
        """Show all segments in the current session."""
        if not self.current_session:
            print("❌ No session loaded.")
            return
        
        print("\n📋 Video Segments")
        print("-" * 30)
        
        for i, segment in enumerate(self.current_session.segments):
            print(f"Segment {i+1}:")
            print(f"  Action: {segment.action_type}")
            print(f"  Clips: {segment.start_clip + 1} to {segment.end_clip + 1} ({segment.clip_count} clips)")
            print(f"  Duration: {segment.clip_count * self.config['video']['clip_duration']:.1f} seconds")
            print(f"  Confidence: {segment.confidence_level:.2f}")
            if segment.is_suspicious:
                print(f"  ⚠️  Suspicious")
            print()
    
    def _get_detailed_description(self):
        """Get detailed description for a segment."""
        if not self.current_session:
            print("❌ No session loaded.")
            return
        
        print("\n📖 Detailed Description")
        print("-" * 30)
        
        # Show available segments
        print("Available segments:")
        for i, segment in enumerate(self.current_session.segments):
            print(f"  {i+1}. {segment.action_type} ({segment.clip_count} clips)")
        
        try:
            segment_idx = input("\nEnter segment number: ").strip()
            if not segment_idx:
                return
            
            segment_idx = int(segment_idx) - 1
            if segment_idx < 0 or segment_idx >= len(self.current_session.segments):
                print("❌ Invalid segment number.")
                return
            
            segment = self.current_session.segments[segment_idx]
            
            print(f"\n📖 Getting detailed description for segment {segment_idx + 1}...")
            
            description_result = self.session_manager.generate_detailed_description(
                self.current_session, segment
            )
            
            if "error" in description_result:
                print(f"❌ Error: {description_result['error']}")
                return
            
            print(f"\nSegment: {segment.action_type}")
            print(f"Duration: {segment.clip_count * self.config['video']['clip_duration']:.1f} seconds")
            print(f"Clips: {segment.start_clip + 1} to {segment.end_clip + 1}")
            print()
            
            if "detailed_description" in description_result:
                print("Detailed Description:")
                print(description_result["detailed_description"])
                print()
            
            if "medical_analysis" in description_result:
                print("Medical Analysis:")
                print(description_result["medical_analysis"])
                print()
            
            if "movement_quality" in description_result:
                print("Movement Quality:")
                print(description_result["movement_quality"])
            
        except ValueError:
            print("❌ Invalid segment number.")
        except Exception as e:
            print(f"❌ Failed to get detailed description: {e}")
    
    def _save_session(self):
        """Save the current session to file."""
        if not self.current_session:
            print("❌ No session loaded.")
            return
        
        print("\n💾 Save Session")
        print("-" * 30)
        
        output_file = input("Enter output file path (default: session.json): ").strip()
        if not output_file:
            output_file = "session.json"
        
        try:
            self.session_manager.save_session(self.current_session, output_file)
            print(f"✅ Session saved to {output_file}")
        except Exception as e:
            print(f"❌ Failed to save session: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Interactive Video Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    
    return parser.parse_args()

def main():
    """Launch the CLI application."""
    args = parse_args()
    
    if args.debug:
        print("🐛 Debug mode enabled")
    
    cli = InteractiveVideoAnalysisCLI()
    cli.run()


if __name__ == "__main__":
    main() 