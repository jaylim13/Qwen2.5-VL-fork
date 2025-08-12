"""
Configuration loader for interactive video analysis.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Use default config path
        config_path = os.path.join(os.path.dirname(__file__), 'analysis_config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'confidence': {
            'temporal_window_size': 5,
            'confidence_threshold': 0.3,
            'strictness_level': 2
        },
        'context': {
            'description_level': 'general',
            'medical_context': True,
            'detail_threshold': 0.7
        },
        'performance': {
            'batch_size': 10,
            'max_clips_per_analysis': 1000
        },
        'actions': [
            'walking', 'sitting', 'standing', 'go upstair', 'go downstair'
        ],
        'model': {
            'name': 'Qwen/Qwen2.5-VL-72B-Instruct',
            'processor_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'max_new_tokens': 32,
            'temperature': 0.1,
            'do_sample': False
        },
        'video': {
            'clip_duration': 2.0,
            'supported_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        }
    } 