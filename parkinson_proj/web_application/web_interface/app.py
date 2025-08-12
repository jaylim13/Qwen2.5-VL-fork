"""
Enhanced Interactive Video Analysis System with all requested features.
Built incrementally to avoid the boolean error.
"""

import os
import sys
import json
import shutil
from datetime import datetime
import gradio as gr
from typing import Dict, Any, Tuple, Optional, List

# Add the parkinson_proj directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from parkinson_proj.web_application.core.session_manager import SessionManager
from parkinson_proj.web_application.config.analysis_config import load_config
from parkinson_proj.web_application.utils.video_utils import VideoSegmentCreator


class EnhancedVideoApp:
    """Enhanced Interactive Video Analysis System."""
    
    def __init__(self):
        """Initialize the web application."""
        self.config = load_config()
        print("🔄 Initializing system and loading model...")
        self.session_manager = SessionManager()
        self.video_segment_creator = VideoSegmentCreator()
        self.current_session = None
        self.current_segment_index = 0
        self.loaded_clips = []
        self.video_folder = ""
        self.segment_videos = {}  # Store segment video paths
        self.transition_videos = {}  # Store transition video paths
        self.temp_video_dir = None  # Temporary directory for video files
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_on_exit)
    
    def cleanup_on_exit(self):
        """Clean up temporary files when the app exits."""
        print("🧹 Cleaning up temporary files...")
        
        # Clean up video segment creator if it exists
        if hasattr(self, 'video_segment_creator') and self.video_segment_creator:
            self.video_segment_creator.cleanup_temp_videos()
        
        # Clean up temp video directory
        if hasattr(self, 'temp_video_dir') and self.temp_video_dir and os.path.exists(self.temp_video_dir):
            try:
                shutil.rmtree(self.temp_video_dir)
                print(f"🧹 Cleaned up temp video directory: {self.temp_video_dir}")
            except Exception as e:
                print(f"⚠️  Error cleaning up temp video directory: {e}")
    
    def create_temp_video_directory(self):
        """Create a temporary directory for video files that Gradio can access."""
        import tempfile
        self.temp_video_dir = tempfile.mkdtemp(prefix="gradio_videos_")
        print(f"📁 Created temp video directory: {self.temp_video_dir}")
        return self.temp_video_dir
    
    def copy_video_to_temp(self, video_path: str) -> str:
        """Copy a video file to the temporary directory for Gradio access."""
        if not self.temp_video_dir:
            self.create_temp_video_directory()
        
        filename = os.path.basename(video_path)
        temp_path = os.path.join(self.temp_video_dir, filename)
        
        try:
            shutil.copy2(video_path, temp_path)
            print(f"📋 Copied {video_path} to {temp_path}")
            return temp_path
        except Exception as e:
            print(f"❌ Error copying video: {e}")
            return video_path  # Fallback to original path
    
    def create_interface(self):
        """Create the enhanced interface with all requested features."""
        
        with gr.Blocks(
            title="Enhanced Interactive Video Analysis System",
            css=self.get_custom_css()
        ) as interface:
            
            gr.Markdown("# 🎥 Enhanced Interactive Video Analysis System")
            
            # Loading Section
            with gr.Row():
                with gr.Column(scale=2):
                    video_folder_input = gr.Textbox(
                        label="Video Folder Path", 
                        value="./data/video_0",
                        placeholder="Enter path to video folder..."
                    )
                with gr.Column(scale=1):
                    max_clips_input = gr.Slider(
                        minimum=10, 
                        maximum=1000, 
                        value=50, 
                        label="Max Clips"
                    )
                
            # Combined Operations
            with gr.Row():
                with gr.Column(scale=2):
                    analyze_button = gr.Button("🎬 Load & Analyze Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    play_transition_button = gr.Button("🔄 Play Transition", variant="secondary", interactive=False)
                with gr.Column(scale=2):
                    analysis_status = gr.Textbox(
                        label="Status",
                        value="Ready to load and analyze video folder...",
                        interactive=False
                    )
            
            # Progress Tracking
            with gr.Row() as progress_section:
                analysis_progress = gr.Progress()
                progress_text = gr.Textbox(
                    label="Analysis Progress",
                    interactive=False,
                    lines=3
                )
            
            # Session Info
            session_info_display = gr.JSON(label="📊 Session Information")
            
            # Main Content Area
            with gr.Row():
                # Video Player (Left Side)
                with gr.Column(scale=3):
                    video_player = gr.Video(
                        label="🎬 Main Video Player (Segments)",
                        interactive=False,
                        show_download_button=False
                    )
                    
                    # Additional info display
                    video_info = gr.HTML(
                        value="<div class='segment-info'>Load a video to start analysis</div>"
                    )
                    
                    # Separate Transition Video Player
                    transition_video_player = gr.Video(
                        label="🔄 Transition Video Player",
                        interactive=False,
                        show_download_button=False,
                        visible=False
                    )
                    
                    # Transition info display
                    transition_info = gr.HTML(
                        value="<div class='segment-info'>No transition video loaded</div>",
                        visible=False
                    )
                    
                    # Timeline Navigation
                    timeline_display = gr.HTML(
                        label="📈 Timeline", 
                        value="<div class='timeline-placeholder'>Timeline will appear here</div>"
                    )
                    
                    # Navigation Controls
                    with gr.Row():
                        prev_segment_btn = gr.Button("⬅️ Previous", variant="secondary")
                        segment_counter = gr.Markdown("**Segment:** - / -")
                        next_segment_btn = gr.Button("➡️ Next", variant="secondary")
                
                # Analysis Panel (Right Side)
                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 Analysis Results")
                    
                    # Structured Analysis Tabs
                    with gr.Tabs() as analysis_tabs:
                        with gr.Tab("👤 Subject"):
                            subject_analysis = gr.Textbox(
                                label="Subject Analysis",
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                placeholder="Subject analysis will appear here...",
                                show_copy_button=True
                            )
                        
                        with gr.Tab("🤝 Interaction"):
                            interaction_analysis = gr.Textbox(
                                label="Interaction Analysis", 
                                lines=10,
                                max_lines=15,
                                interactive=False,
                                placeholder="Interaction analysis will appear here...",
                                show_copy_button=True
                            )
                        
                        with gr.Tab("🌍 Context"):
                            context_analysis = gr.Textbox(
                                label="Context Analysis",
                                lines=10,
                                max_lines=15, 
                                interactive=False,
                                placeholder="Context analysis will appear here...",
                                show_copy_button=True
                            )
                        
                        with gr.Tab("📋 Summary"):
                            segment_summary = gr.Textbox(
                                label="Segment Summary",
                                lines=8,
                                max_lines=12,
                                interactive=False,
                                placeholder="Segment summary will appear here...",
                                show_copy_button=True
                            )
            
            # Export Section
            with gr.Row():
                with gr.Column():
                    export_format = gr.Dropdown(
                        choices=["JSON", "CSV", "TXT", "HTML"],
                        value="JSON",
                        label="Export Format"
                    )
                with gr.Column():
                    export_button = gr.Button("📤 Export Analysis", variant="secondary")
                with gr.Column():
                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False,
                        placeholder="Ready to export..."
                    )
            
            # Event Handlers
            
            # Combined load and analysis processing
            analyze_button.click(
                fn=self.load_and_analyze_video,
                inputs=[video_folder_input, max_clips_input],
                outputs=[
                    analysis_status,
                    video_player,
                    video_info,
                    timeline_display,
                    segment_counter,
                    subject_analysis,
                    interaction_analysis,
                    context_analysis,
                    segment_summary,
                    progress_section,
                    play_transition_button,
                    session_info_display
                ]
            )

            # Transition playback handler - now uses separate player
            play_transition_button.click(
                fn=self.play_current_transition,
                inputs=[],
                outputs=[transition_video_player, transition_info]
            )
            
            # Navigation handlers
            prev_segment_btn.click(
                fn=self.navigate_to_previous,
                inputs=[],
                outputs=[
                    video_player,
                    video_info,
                    timeline_display, 
                    segment_counter,
                    subject_analysis,
                    interaction_analysis,
                    context_analysis,
                    segment_summary
                ]
            )
            
            next_segment_btn.click(
                fn=self.navigate_to_next,
                inputs=[],
                outputs=[
                    video_player,
                    video_info,
                    timeline_display,
                    segment_counter, 
                    subject_analysis,
                    interaction_analysis,
                    context_analysis,
                    segment_summary
                ]
            )
            
            # Export handler
            export_button.click(
                fn=self.export_analysis,
                inputs=[export_format],
                outputs=[export_status]
            )
        
        return interface
    
    def get_custom_css(self) -> str:
        """Get custom CSS for enhanced UI."""
        return """
        .video-placeholder {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 20px;
            text-align: center;
            border-radius: 10px;
            font-size: 18px;
            margin: 10px 0;
        }
        
        .timeline-placeholder {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            color: #6c757d;
            margin: 10px 0;
        }
        
        .timeline-bar {
            background: #e9ecef;
            height: 20px;
            border-radius: 10px;
            position: relative;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-high { background-color: #28a745; }
        .confidence-medium { background-color: #ffc107; }
        .confidence-low { background-color: #dc3545; }
        
        .video-player-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .segment-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .analysis-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .export-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        """
    
    def load_and_analyze_video(self, folder: str, max_clips: int):
        """Complete workflow: Load folder, predict clips, analyze segments, and return full results."""
        try:
            if not os.path.exists(folder):
                error_result = (
                    "❌ Video folder not found",
                    None, "<div class='segment-info'>Error loading videos</div>",
                    "<div class='timeline-placeholder'>Error</div>", "**Error**",
                    "No analysis available", "No analysis available", "No analysis available", "No analysis available",
                    gr.update(visible=False), gr.update(interactive=False), {"error": "Folder not found"}
                )
                return error_result
            
            # Step 1: Quick file scanning
            print("📁 Scanning video folder...")
            import glob
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
                video_files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
            
            if not video_files:
                error_result = (
                    "❌ No video files found",
                    None, "<div class='segment-info'>No video files found</div>",
                    "<div class='timeline-placeholder'>No videos</div>", "**Error**",
                    "No analysis available", "No analysis available", "No analysis available", "No analysis available",
                    gr.update(visible=False), gr.update(interactive=False), {"error": "No videos"}
                )
                return error_result
            
            # Limit clips
            video_files = sorted(video_files)[:max_clips]
            
            # Step 2: Run clip predictions
            print("🎬 Running video clip predictions...")
            annotated_clips = self.session_manager.video_annotator.annotate_video_folder(folder, max_clips)
            
            # Step 3: Calculate confidence scores
            print("🔍 Calculating confidence scores...")
            from utils.confidence_utils import update_clip_confidence_scores
            annotated_clips = update_clip_confidence_scores(annotated_clips, self.config)
            
            # Step 4: Create segments from clips
            print("📊 Creating segments from clips...")
            segments = self.session_manager.context_analyzer.create_segments_from_clips(annotated_clips)
            
            # Step 5: Create segment and transition videos
            print("🎥 Creating segment videos...")
            self.segment_videos = self.video_segment_creator.create_segment_videos(annotated_clips, segments)
            
            print("🔄 Creating transition videos...")
            self.transition_videos = self.video_segment_creator.create_transition_videos(annotated_clips, segments)
            
            # Step 6: Update segments with video paths
            for i, segment in enumerate(segments):
                segment_key = f"segment_{i}"
                if segment_key in self.segment_videos:
                    segment.segment_video_path = self.segment_videos[segment_key]
                
                # Add transition video path if available
                if i < len(segments) - 1:
                    transition_key = f"transition_{i}"
                    if transition_key in self.transition_videos:
                        segment.transition_video_path = self.transition_videos[transition_key]
            
            # Step 7: Generate detailed analyses
            print("🧠 Generating detailed analysis...")
            analyzed_segments = self.session_manager.context_analyzer.analyze_all_segments(
                annotated_clips, description_level="detailed"
            )
            
            # Keep video paths we created
            for i, seg in enumerate(analyzed_segments):
                if i < len(segments):
                    seg.segment_video_path = segments[i].segment_video_path
                    seg.transition_video_path = segments[i].transition_video_path
            
            # Step 8: Create session with full analysis
            from utils.data_structures import AnalysisSession
            self.current_session = AnalysisSession(
                video_folder=folder,
                annotated_clips=annotated_clips,
                segments=analyzed_segments,
                confidence_level=2,
                description_level="detailed",
                medical_context=True
            )
            
            self.current_segment_index = 0
            
            # Step 9: Generate UI components with full analysis
            timeline_html = self._generate_timeline()
            counter_text = f"**Segment:** 1 / {len(analyzed_segments)}"
            
            # Get current segment analysis
            if analyzed_segments:
                current_segment = analyzed_segments[0]
                analysis_text = current_segment.general_description + " " + current_segment.detailed_description
                subject, interaction, context, summary = self._parse_analysis(analysis_text)
            else:
                subject = interaction = context = summary = "No segments available"
            
            # Get current video path and info for display
            if analyzed_segments and len(analyzed_segments) > 0:
                current_segment = analyzed_segments[0]
                # Use segment video path if available
                if current_segment.segment_video_path and os.path.exists(current_segment.segment_video_path):
                    current_video_path = self.copy_video_to_temp(current_segment.segment_video_path)
                    video_info_html = f"""
                    <div class="segment-info">
                        <strong>Segment 1</strong> | 
                        Action: {current_segment.action_type} | 
                        Clips: {current_segment.clip_count} | 
                        Confidence: {current_segment.confidence_level:.2f}
                        <br><small>Showing combined segment video ({current_segment.clip_count} clips)</small>
                    </div>
                    """
                else:
                    # Fallback to first clip
                    original_video_path = annotated_clips[0].video_path
                    current_video_path = self.copy_video_to_temp(original_video_path)
                    video_info_html = f"""
                    <div class="segment-info">
                        <strong>Segment 1</strong> | 
                        Action: {current_segment.action_type} | 
                        Clips: {current_segment.clip_count} | 
                        Confidence: {current_segment.confidence_level:.2f}
                        <br><small>Showing first clip (segment video creation failed)</small>
                    </div>
                    """
            else:
                current_video_path = None
                video_info_html = "<div class='segment-info'>No segments available</div>"
            
            # Session info
            session_info = {
                "folder": folder,
                "total_clips": len(annotated_clips),
                "total_segments": len(analyzed_segments),
                "status": "complete",
                "timestamp": datetime.now().isoformat()
            }
            
            return (
                f"✅ Complete! Loaded {len(annotated_clips)} clips and created {len(analyzed_segments)} segments with full analysis.",
                current_video_path,
                video_info_html,
                timeline_html,
                counter_text,
                subject,
                interaction, 
                context,
                summary,
                gr.update(visible=True),  # Show progress section
                gr.update(interactive=True),  # Enable transition button
                session_info
            )
            
        except Exception as e:
            print(f"❌ Error in load_and_analyze_video: {e}")
            error_result = (
                f"❌ Error: {str(e)}",
                None, "<div class='segment-info'>Analysis failed</div>",
                "<div class='timeline-placeholder'>Analysis failed</div>", "**Error**",
                f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}", f"Error: {str(e)}",
                gr.update(visible=False), gr.update(interactive=False), {"error": str(e)}
            )
            return error_result
    
    
    def navigate_to_previous(self):
        """Navigate to previous segment."""
        if not self.current_session or not self.current_session.segments:
            return self._return_navigation_error()
        
        if self.current_segment_index > 0:
            self.current_segment_index -= 1
        
        return self._get_current_segment_display()
    
    def navigate_to_next(self):
        """Navigate to next segment."""
        if not self.current_session or not self.current_session.segments:
            return self._return_navigation_error()
        
        if self.current_segment_index < len(self.current_session.segments) - 1:
            self.current_segment_index += 1
        
        return self._get_current_segment_display()
    
    def export_analysis(self, format_type: str) -> str:
        """Export analysis in specified format."""
        if not self.current_session:
            return "❌ No session loaded"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_export_{timestamp}.{format_type.lower()}"
            filepath = os.path.join("output", filename)
            
            # Ensure output directory exists
            os.makedirs("output", exist_ok=True)
            
            if format_type == "JSON":
                self._export_json(filepath)
            elif format_type == "CSV":
                self._export_csv(filepath)
            elif format_type == "TXT":
                self._export_txt(filepath)
            elif format_type == "HTML":
                self._export_html(filepath)
            
            return f"✅ Exported to {filepath}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def _generate_video_player(self) -> str:
        """Generate enhanced video player HTML."""
        if not self.current_session or not self.current_session.segments:
            return "<div class='video-placeholder'>No video segments available</div>"
        
        current_segment = self.current_session.segments[self.current_segment_index]
        
        # Get the first clip from the current segment to display
        if current_segment.start_clip < len(self.current_session.annotated_clips):
            first_clip = self.current_session.annotated_clips[current_segment.start_clip]
            video_path = first_clip.video_path
        else:
            return "<div class='video-placeholder'>No video available for this segment</div>"
        
        return f"""
        <div class="video-player-container">
            <video controls width="100%" style="border-radius: 10px;">
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="segment-info">
                <strong>Segment {self.current_segment_index + 1}</strong> | 
                Action: {current_segment.action_type} | 
                Clips: {current_segment.clip_count} | 
                Confidence: {current_segment.confidence_level:.2f}
            </div>
        </div>
        """
    
    def _generate_timeline(self) -> str:
        """Generate color-coded timeline."""
        if not self.current_session or not self.current_session.segments:
            return "<div class='timeline-placeholder'>Timeline not available</div>"
        
        segments = self.current_session.segments
        timeline_html = "<div class='timeline-bar'>"
        
        for i, segment in enumerate(segments):
            confidence_class = self._get_confidence_class(segment.confidence_level)
            width_percent = 100 / len(segments)
            selected_class = "selected" if i == self.current_segment_index else ""
            
            timeline_html += f"""
            <div class="timeline-segment {confidence_class} {selected_class}" 
                 style="width: {width_percent}%; height: 100%; float: left; 
                        border-left: 1px solid #fff;"
                 title="Segment {i+1}: {segment.action_type} - Confidence {segment.confidence_level:.2f}">
            </div>
            """
        
        timeline_html += "</div>"
        timeline_html += """
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px;">
            <span>🟢 High Confidence</span>
            <span>🟡 Medium Confidence</span>
            <span>🔴 Low Confidence</span>
        </div>
        """
        
        return timeline_html
    
    def _generate_clips_timeline(self) -> str:
        """Generate timeline from individual clips (before segmentation)."""
        if not self.current_session or not self.current_session.annotated_clips:
            return "<div class='timeline-placeholder'>No clips loaded</div>"
        
        clips = self.current_session.annotated_clips
        timeline_html = "<div class='timeline-bar'>"
        
        for i, clip in enumerate(clips):
            confidence_class = self._get_confidence_class(clip.confidence_score)
            width_percent = 100 / len(clips)
            
            timeline_html += f"""
            <div class="timeline-segment {confidence_class}" 
                 style="width: {width_percent}%; height: 100%; float: left; 
                        border-left: 1px solid #fff;"
                 title="Clip {i+1}: {clip.predicted_action} - Confidence {clip.confidence_score:.2f}">
            </div>
            """
        
        timeline_html += "</div>"
        timeline_html += """
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px;">
            <span>🟢 High Confidence</span>
            <span>🟡 Medium Confidence</span>
            <span>🔴 Low Confidence</span>
        </div>
        """
        
        return timeline_html
    
    def _generate_basic_video_player(self) -> str:
        """Generate basic video player from first clip (before segmentation)."""
        if not self.current_session or not self.current_session.annotated_clips:
            return "<div class='video-placeholder'>No clips available</div>"
        
        first_clip = self.current_session.annotated_clips[0]
        
        return f"""
        <div class="video-player-container">
            <video controls width="100%" style="border-radius: 10px;">
                <source src="{first_clip.video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="segment-info">
                <strong>First Clip</strong> | 
                Action: {first_clip.predicted_action} | 
                Confidence: {first_clip.confidence_score:.2f}
                <br><small>Showing individual clips - click "Analyze Video" for segments</small>
            </div>
        </div>
        """
    
    def _parse_analysis(self, analysis_text: str) -> Tuple[str, str, str, str]:
        """Parse analysis text into structured components."""
        if not analysis_text or len(analysis_text.strip()) < 10:
            return ("No subject analysis available", 
                   "No interaction analysis available", 
                   "No context analysis available",
                   "No analysis available")
        
        lines = analysis_text.split('\n')
        
        subject = ""
        interaction = ""
        context = ""
        current_section = None
        
        # Parse structured output from the new prompt format
        for line in lines:
            line = line.strip()
            
            if "# 1. SUBJECT:" in line.upper() or "1. SUBJECT:" in line.upper():
                current_section = "subject"
                continue
            elif "# 2. INTERACTION:" in line.upper() or "2. INTERACTION:" in line.upper():
                current_section = "interaction"
                continue
            elif "# 3. CONTEXT:" in line.upper() or "3. CONTEXT:" in line.upper():
                current_section = "context"
                continue
            elif line.startswith("#") and any(x in line.upper() for x in ["SUBJECT", "INTERACTION", "CONTEXT"]):
                # Handle other section headers
                if "SUBJECT" in line.upper():
                    current_section = "subject"
                elif "INTERACTION" in line.upper():
                    current_section = "interaction"
                elif "CONTEXT" in line.upper():
                    current_section = "context"
                continue
            
            # Add content to current section - be very permissive, only skip obvious template/metadata lines
            if current_section == "subject" and line:
                # Skip only obvious template lines, not actual content
                if not (line.strip().startswith("[Describe what the subject") or 
                       line.strip().startswith("Segment:") or 
                       line.strip().startswith("Action:") or 
                       line.strip().startswith("Duration:") or
                       line.strip().startswith("Please follow the exact format")):
                    subject += line + " "
            elif current_section == "interaction" and line:
                if not (line.strip().startswith("[Describe whether the subject") or 
                       line.strip().startswith("Segment:") or 
                       line.strip().startswith("Action:") or 
                       line.strip().startswith("Duration:") or
                       line.strip().startswith("Please follow the exact format")):
                    interaction += line + " "
            elif current_section == "context" and line:
                if not (line.strip().startswith("[Describe the environment") or 
                       line.strip().startswith("Segment:") or 
                       line.strip().startswith("Action:") or 
                       line.strip().startswith("Duration:") or
                       line.strip().startswith("Please follow the exact format")):
                    context += line + " "
        
        # Clean up content and remove artifacts
        def clean_content(content):
            if not content:
                return content
            # Remove trailing artifacts like "#", "# 3", etc.
            content = content.strip()
            # Remove standalone # symbols at the end
            while content.endswith("#") or content.endswith("# "):
                content = content.rstrip("# ").strip()
            # Remove numbered artifacts like "# 3", "# 2", etc.
            import re
            content = re.sub(r'\s*#\s*\d*\s*$', '', content).strip()
            return content
        
        subject = clean_content(subject) if clean_content(subject) else "Subject analysis not clearly structured in model response"
        interaction = clean_content(interaction) if clean_content(interaction) else "Interaction analysis not clearly structured in model response"
        context = clean_content(context) if clean_content(context) else "Context analysis not clearly structured in model response"
        
        # Create summary from all sections - allow much longer summaries
        all_content = f"{subject} {interaction} {context}".strip()
        summary = all_content[:1000] + "..." if len(all_content) > 1000 else all_content
        if not summary.strip():
            summary = analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text
        
        return subject, interaction, context, summary
    
    def _get_current_segment_display(self):
        """Get display data for current segment."""
        if not self.current_session or not self.current_session.segments:
            return self._return_navigation_error()
        
        # Get video path and info
        current_segment = self.current_session.segments[self.current_segment_index]
        
        # Use segment video path if available, otherwise fall back to first clip
        if current_segment.segment_video_path and os.path.exists(current_segment.segment_video_path):
            video_path = self.copy_video_to_temp(current_segment.segment_video_path)
            video_info_html = f"""
            <div class="segment-info">
                <strong>Segment {self.current_segment_index + 1}</strong> | 
                Action: {current_segment.action_type} | 
                Clips: {current_segment.clip_count} | 
                Confidence: {current_segment.confidence_level:.2f}
                <br><small>Showing combined segment video ({current_segment.clip_count} clips)</small>
            </div>
            """
        elif current_segment.start_clip < len(self.current_session.annotated_clips):
            original_video_path = self.current_session.annotated_clips[current_segment.start_clip].video_path
            video_path = self.copy_video_to_temp(original_video_path)
            video_info_html = f"""
            <div class="segment-info">
                <strong>Segment {self.current_segment_index + 1}</strong> | 
                Action: {current_segment.action_type} | 
                Clips: {current_segment.clip_count} | 
                Confidence: {current_segment.confidence_level:.2f}
                <br><small>Showing first clip (segment video creation failed)</small>
            </div>
            """
        else:
            video_path = None
            video_info_html = "<div class='segment-info'>No video available</div>"
        
        timeline_html = self._generate_timeline()
        counter_text = f"**Segment:** {self.current_segment_index + 1} / {len(self.current_session.segments)}"
        
        analysis_text = current_segment.general_description + " " + current_segment.detailed_description
        subject, interaction, context, summary = self._parse_analysis(analysis_text)
        
        return (video_path, video_info_html, timeline_html, counter_text, subject, interaction, context, summary)

    def play_current_transition(self):
        """Return the transition video for the current segment if available."""
        if not self.current_session or not self.current_session.segments:
            return gr.update(visible=False), gr.update(visible=False, value="<div class='segment-info'>No session loaded</div>")
        
        idx = self.current_segment_index
        if idx >= len(self.current_session.segments):
            return gr.update(visible=False), gr.update(visible=False, value="<div class='segment-info'>No segment selected</div>")
        
        seg = self.current_session.segments[idx]
        
        # Check if transition video exists
        if hasattr(seg, 'transition_video_path') and seg.transition_video_path and os.path.exists(seg.transition_video_path):
            path = self.copy_video_to_temp(seg.transition_video_path)
            next_action = self.current_session.segments[idx+1].action_type if idx+1 < len(self.current_session.segments) else 'End'
            
            info = f"""
            <div class='segment-info'>
                <strong>🔄 Transition Video: Segment {idx+1} → Segment {idx+2}</strong><br>
                From: <strong>{seg.action_type}</strong> → To: <strong>{next_action}</strong><br>
                <small>Shows last clip of current segment + first clip of next segment</small>
            </div>
            """
            return gr.update(value=path, visible=True), gr.update(value=info, visible=True)
        else:
            # No transition available for this segment
            if idx >= len(self.current_session.segments) - 1:
                info = "<div class='segment-info'>⚠️ No transition video available (this is the last segment)</div>"
            else:
                info = "<div class='segment-info'>⚠️ Transition video not found for this segment</div>"
            
            return gr.update(visible=False), gr.update(value=info, visible=True)
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class based on confidence level."""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _return_error(self, message: str):
        """Return error state for all outputs."""
        error_html = f"<div class='video-placeholder'>❌ {message}</div>"
        return (
            {"error": message},
            error_html,
            f"<div class='timeline-placeholder'>❌ {message}</div>",
            "**Error**",
            f"Error: {message}",
            f"Error: {message}",
            f"Error: {message}",
            f"Error: {message}"
        )
    
    def _return_navigation_error(self):
        """Return navigation error state."""
        return (
            None,  # No video file
            "<div class='segment-info'>No session loaded</div>",
            "<div class='timeline-placeholder'>No session loaded</div>",
            "**Segment:** - / -",
            "No session loaded",
            "No session loaded", 
            "No session loaded",
            "No session loaded"
        )
    
    def _export_json(self, filepath: str):
        """Export analysis as JSON with complete, non-truncated content."""
        data = {
            "session_info": {
                "total_segments": len(self.current_session.segments),
                "export_timestamp": datetime.now().isoformat()
            },
            "segments": []
        }
        
        for i, segment in enumerate(self.current_session.segments):
            # Parse analysis but preserve complete content - avoid truncation
            combined_analysis = segment.general_description + " " + segment.detailed_description
            subject, interaction, context, summary = self._parse_analysis(combined_analysis)
            
            # If parsing failed to extract structured content, include the raw analysis
            if len(subject) < 50 or len(interaction) < 50 or len(context) < 50:
                print(f"⚠️  Warning: Segment {i} may have parsing issues, including raw content as backup")
                
                # Try to extract better content or use raw descriptions
                raw_subject = segment.general_description if segment.general_description else "No general description available"
                raw_interaction = segment.detailed_description if segment.detailed_description else "No detailed description available" 
                raw_context = combined_analysis[:500] if len(combined_analysis) > 500 else combined_analysis
                raw_summary = combined_analysis
                
                data["segments"].append({
                    "index": i,
                    "action_type": segment.action_type,
                    "confidence": segment.confidence_level,
                    "clip_count": segment.clip_count,
                    "analysis": {
                        "subject": subject if len(subject) > 50 else raw_subject,
                        "interaction": interaction if len(interaction) > 50 else raw_interaction,
                        "context": context if len(context) > 50 else raw_context,
                        "summary": raw_summary  # Always use full content for summary
                    },
                    "raw_content": {
                        "general_description": segment.general_description,
                        "detailed_description": segment.detailed_description,
                        "combined_analysis": combined_analysis
                    }
                })
            else:
                data["segments"].append({
                    "index": i,
                    "action_type": segment.action_type,
                    "confidence": segment.confidence_level,
                    "clip_count": segment.clip_count,
                    "analysis": {
                        "subject": subject,
                        "interaction": interaction,
                        "context": context,
                        "summary": summary
                    },
                    "raw_content": {
                        "general_description": segment.general_description,
                        "detailed_description": segment.detailed_description
                    }
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, filepath: str):
        """Export analysis as CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Segment', 'Confidence', 'Subject', 'Interaction', 'Context', 'Summary'])
            
            for i, segment in enumerate(self.current_session.segments):
                subject, interaction, context, summary = self._parse_analysis(segment.general_description + " " + segment.detailed_description)
                writer.writerow([i+1, segment.confidence_level, subject, interaction, context, summary])
    
    def _export_txt(self, filepath: str):
        """Export analysis as TXT."""
        with open(filepath, 'w') as f:
            f.write("Interactive Video Analysis Export\n")
            f.write("=" * 40 + "\n\n")
            
            for i, segment in enumerate(self.current_session.segments):
                subject, interaction, context, summary = self._parse_analysis(segment.general_description + " " + segment.detailed_description)
                f.write(f"SEGMENT {i+1}\n")
                f.write(f"Action: {segment.action_type}\n")
                f.write(f"Confidence: {segment.confidence_level:.2f}\n")
                f.write(f"Clips: {segment.clip_count}\n")
                f.write(f"Subject: {subject}\n")
                f.write(f"Interaction: {interaction}\n") 
                f.write(f"Context: {context}\n")
                f.write(f"Summary: {summary}\n")
                f.write("-" * 40 + "\n\n")
    
    def _export_html(self, filepath: str):
        """Export analysis as HTML."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Analysis Export</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .segment { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .confidence-high { border-left: 5px solid #28a745; }
                .confidence-medium { border-left: 5px solid #ffc107; }
                .confidence-low { border-left: 5px solid #dc3545; }
            </style>
        </head>
        <body>
            <h1>🎥 Interactive Video Analysis Export</h1>
        """
        
        for i, segment in enumerate(self.current_session.segments):
            confidence_class = self._get_confidence_class(segment.confidence_level).replace('confidence-', 'confidence-')
            subject, interaction, context, summary = self._parse_analysis(segment.general_description + " " + segment.detailed_description)
            
            html_content += f"""
            <div class="segment {confidence_class}">
                <h3>Segment {i+1}</h3>
                <p><strong>Action:</strong> {segment.action_type}</p>
                <p><strong>Confidence:</strong> {segment.confidence_level:.2f}</p>
                <p><strong>Clips:</strong> {segment.clip_count}</p>
                <p><strong>Subject:</strong> {subject}</p>
                <p><strong>Interaction:</strong> {interaction}</p>
                <p><strong>Context:</strong> {context}</p>
                <p><strong>Summary:</strong> {summary}</p>
            </div>
            """
        
        html_content += "</body></html>"
        
        with open(filepath, 'w') as f:
            f.write(html_content)


def main():
    """Launch the enhanced web application."""
    app = EnhancedVideoApp()
    
    # Create temp directory before launching
    app.create_temp_video_directory()
    
    interface = app.create_interface()
    
    print("🚀 Starting Enhanced Interactive Video Analysis System")
    print("=" * 60)
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        allowed_paths=[app.temp_video_dir] if app.temp_video_dir else []
    )


if __name__ == "__main__":
    main()