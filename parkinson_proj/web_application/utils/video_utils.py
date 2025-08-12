"""
Video utilities for segment creation and video concatenation.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import atexit

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

class VideoSegmentCreator:
    """Handles creation of segment videos and transition videos."""
    
    def __init__(self):
        """Initialize the video segment creator."""
        self.temp_dir = None
        self.created_videos = []
        
        # Register cleanup on exit
        atexit.register(self.cleanup_temp_videos)
    
    def create_temp_directory(self, session_id: str = None) -> str:
        """Create a temporary directory for video files."""
        if session_id is None:
            session_id = "default"
        
        self.temp_dir = f"/tmp/video_analysis_{session_id}"
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"📁 Created temp directory: {self.temp_dir}")
        return self.temp_dir
    
    def concatenate_videos(self, video_paths: List[str], output_path: str) -> bool:
        """Concatenate multiple video files into one."""
        if not video_paths:
            return False
        
        # Check if ffmpeg is available
        if not self.check_ffmpeg_available():
            print("⚠️  FFmpeg not available - attempting OpenCV fallback for concatenation")
            return self._concatenate_videos_opencv(video_paths, output_path)
        
        try:
            # Create a file list for ffmpeg with absolute paths
            file_list_path = os.path.join(self.temp_dir, "file_list.txt")
            with open(file_list_path, 'w') as f:
                for video_path in video_paths:
                    # Convert to absolute path for ffmpeg
                    abs_path = os.path.abspath(video_path)
                    f.write(f"file '{abs_path}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', file_list_path,
                '-c', 'copy',  # Copy without re-encoding for speed
                output_path,
                '-y'  # Overwrite output file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Created segment video: {output_path}")
                self.created_videos.append(output_path)
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                # Try OpenCV fallback
                print("➡️  Trying OpenCV fallback for concatenation...")
                return self._concatenate_videos_opencv(video_paths, output_path)
                
        except Exception as e:
            print(f"❌ Error concatenating videos: {e}")
            # Try OpenCV fallback
            return self._concatenate_videos_opencv(video_paths, output_path)
    
    def create_segment_videos(self, annotated_clips: List, segments: List) -> Dict[str, str]:
        """Create segment videos from annotated clips."""
        if not self.temp_dir:
            self.create_temp_directory()
        
        segment_videos = {}
        
        for i, segment in enumerate(segments):
            # Get clip paths for this segment
            clip_paths = []
            for clip_idx in range(segment.start_clip, segment.end_clip + 1):
                if clip_idx < len(annotated_clips):
                    clip_paths.append(annotated_clips[clip_idx].video_path)
            
            if clip_paths:
                # Create segment video
                segment_video_path = os.path.join(self.temp_dir, f"segment_{i}_{segment.action_type}.mp4")
                if self.concatenate_videos(clip_paths, segment_video_path):
                    segment_videos[f"segment_{i}"] = segment_video_path
        
        return segment_videos
    
    def create_transition_videos(self, annotated_clips: List, segments: List) -> Dict[str, str]:
        """Create transition videos between different action segments."""
        if not self.temp_dir:
            self.create_temp_directory()
        
        transition_videos = {}
        
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            # Get last clip of current segment and first clip of next segment
            transition_clips = []
            
            # Add last clip of current segment
            if current_segment.end_clip < len(annotated_clips):
                transition_clips.append(annotated_clips[current_segment.end_clip].video_path)
            
            # Add first clip of next segment
            if next_segment.start_clip < len(annotated_clips):
                transition_clips.append(annotated_clips[next_segment.start_clip].video_path)
            
            if len(transition_clips) >= 2:
                # Create transition video
                transition_video_path = os.path.join(
                    self.temp_dir, 
                    f"transition_{i}_{current_segment.action_type}_to_{next_segment.action_type}.mp4"
                )
                if self.concatenate_videos(transition_clips, transition_video_path):
                    transition_videos[f"transition_{i}"] = transition_video_path
        
        return transition_videos
    
    def _concatenate_videos_opencv(self, video_paths: List[str], output_path: str) -> bool:
        """Fallback concatenation using OpenCV re-encoding (MP4V)."""
        if not CV2_AVAILABLE:
            print("❌ OpenCV not available; cannot concatenate videos without ffmpeg.")
            return False
        try:
            # Find first readable video to get properties
            cap0 = None
            for p in video_paths:
                cap0 = cv2.VideoCapture(p)
                if cap0.isOpened():
                    break
                cap0.release()
                cap0 = None
            if cap0 is None:
                print("❌ Could not open any input videos for concatenation")
                return False
            fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap0.release()

            # Ensure parent dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print("❌ Failed to open VideoWriter for output")
                return False

            total_frames = 0
            for path in video_paths:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print(f"⚠️  Skipping unreadable video: {path}")
                    continue
                # Resize frames if needed
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
                    total_frames += 1
                cap.release()

            writer.release()
            if total_frames == 0:
                print("❌ No frames written during OpenCV concatenation")
                return False
            print(f"✅ Created segment video with OpenCV: {output_path} ({total_frames} frames)")
            self.created_videos.append(output_path)
            return True
        except Exception as e:
            print(f"❌ OpenCV concatenation failed: {e}")
            return False

    def cleanup_temp_videos(self):
        """Clean up temporary video files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️  Error cleaning up temp directory: {e}")
    
    def get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video file in seconds."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                return 0.0
        except Exception as e:
            print(f"❌ Error getting video duration: {e}")
            return 0.0
    
    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
