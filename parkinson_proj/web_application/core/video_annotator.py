"""
Video annotation module for interactive video analysis.
Adapts existing Qwen2.5-VL analysis for the interactive system.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - model-based analysis will be skipped")

# Import from parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'qwen-vl-utils', 'src'))

if TORCH_AVAILABLE:
    from qwen_vl_utils import process_vision_info

from parkinson_proj.web_application.utils.data_structures import AnnotatedClip
from parkinson_proj.web_application.prompts.general_prompts import ACTION_CLASSIFICATION_PROMPT
from parkinson_proj.web_application.config.analysis_config import load_config
from parkinson_proj.web_application.core.model_manager import model_manager


class VideoAnnotator:
    """Handles video annotation using Qwen2.5-VL."""
    
    def __init__(self, config_path: str = None):
        """Initialize the video annotator."""
        self.config = load_config(config_path)
        self.actions = self.config['actions']
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
    
    def get_video_files(self, folder_path: str) -> List[str]:
        """Get all video files from the folder."""
        video_extensions = self.config['video']['supported_extensions']
        video_files = []
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder {folder_path} does not exist!")
            return video_files
        
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in video_extensions:
                video_files.append(str(file))
        
        video_files.sort()  # Sort for consistent ordering
        print(f"📁 Found {len(video_files)} video files")
        return video_files
    
    def classify_video_action(self, video_path: str) -> Dict:
        """Classify action in a single video clip."""
        model = model_manager.get_model()
        processor = model_manager.get_processor()
        
        if not TORCH_AVAILABLE or not model or not processor:
            return {
                "video_path": video_path,
                "raw_response": "model_not_available",
                "classified_action": "unknown",
                "confidence": "low"
            }
        
        # Create the prompt for action classification
        prompt = ACTION_CLASSIFICATION_PROMPT.format(actions=', '.join(self.actions))
        
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
        
        try:
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
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=self.config['model']['max_new_tokens'],
                    do_sample=self.config['model']['do_sample'],
                    temperature=self.config['model']['temperature']
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Clean up the response
            action = output_text[0].strip().lower()
            
            # Validate if the action is in our predefined list
            valid_action = None
            for predefined_action in self.actions:
                if predefined_action in action:
                    valid_action = predefined_action
                    break
            
            if valid_action is None:
                # If no exact match, try to find the closest match
                for predefined_action in self.actions:
                    if any(word in action for word in predefined_action.split()):
                        valid_action = predefined_action
                        break
            
            return {
                "video_path": video_path,
                "raw_response": action,
                "classified_action": valid_action or "unknown",
                "confidence": "high" if valid_action else "low"
            }
            
        except Exception as e:
            return {
                "video_path": video_path,
                "raw_response": "",
                "classified_action": "error",
                "confidence": "error",
                "error": str(e)
            }
    
    def annotate_video_folder(self, folder_path: str, max_clips: Optional[int] = None) -> List[AnnotatedClip]:
        """Annotate all video clips in a folder."""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available - creating mock annotations for testing")
            return self._create_mock_annotations(folder_path, max_clips)
        
        # Ensure model is loaded
        if not model_manager.is_model_loaded():
            if not self.load_model():
                return []
        
        # Get video files
        video_files = self.get_video_files(folder_path)
        if not video_files:
            print("❌ No video files found!")
            return []
        
        if max_clips:
            video_files = video_files[:max_clips]
        
        # Annotate each video
        annotated_clips = []
        total_files = len(video_files)
        
        print(f"\n🎬 Annotating {total_files} video clips...")
        print("=" * 50)
        
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            print(f"\n📹 Processing {i}/{total_files}: {video_name}")
            
            start_time = time.time()
            result = self.classify_video_action(video_path)
            end_time = time.time()
            
            # Create AnnotatedClip object
            clip_id = Path(video_path).stem
            timestamp = (i - 1) * self.config['video']['clip_duration']
            
            annotated_clip = AnnotatedClip(
                clip_id=clip_id,
                video_path=video_path,
                timestamp=timestamp,
                predicted_action=result["classified_action"],
                raw_response=result["raw_response"],
                error_message=result.get("error")
            )
            
            annotated_clips.append(annotated_clip)
            
            # Print result
            action = result["classified_action"]
            confidence = result["confidence"]
            print(f"   Action: {action} (confidence: {confidence})")
            print(f"   Time: {end_time - start_time:.2f}s")
            
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        print(f"\n✅ Annotation complete! Processed {len(annotated_clips)} clips.")
        return annotated_clips
    
    def _create_mock_annotations(self, folder_path: str, max_clips: Optional[int] = None) -> List[AnnotatedClip]:
        """Create mock annotations for testing when model is not available."""
        print("🧪 Creating mock annotations for testing...")
        
        # Get video files
        video_files = self.get_video_files(folder_path)
        if not video_files:
            print("❌ No video files found!")
            return []
        
        if max_clips:
            video_files = video_files[:max_clips]
        
        # Create mock annotations
        annotated_clips = []
        actions = ["sitting", "walking", "standing", "sitting", "walking"]
        
        for i, video_path in enumerate(video_files):
            clip_id = Path(video_path).stem
            timestamp = i * self.config['video']['clip_duration']
            
            # Cycle through actions for mock data
            action = actions[i % len(actions)]
            
            annotated_clip = AnnotatedClip(
                clip_id=clip_id,
                video_path=video_path,
                timestamp=timestamp,
                predicted_action=action,
                raw_response=action,
                confidence_score=0.8,
                temporal_consistency=0.7
            )
            
            annotated_clips.append(annotated_clip)
        
        print(f"✅ Mock annotations created! Processed {len(annotated_clips)} clips.")
        return annotated_clips
    
    def save_annotations(self, annotated_clips: List[AnnotatedClip], output_file: str):
        """Save annotations to JSON file."""
        data = [clip.to_dict() for clip in annotated_clips]
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Annotations saved to {output_file}")
    
    def load_annotations(self, input_file: str) -> List[AnnotatedClip]:
        """Load annotations from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        return [AnnotatedClip.from_dict(clip_data) for clip_data in data] 