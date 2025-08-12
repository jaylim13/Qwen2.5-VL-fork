"""
Model Manager for Qwen2.5-VL
Singleton pattern to load model once and reuse across all components.
"""
import os
import sys
from typing import Optional, Dict, Any
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from parkinson_proj.web_application.config.analysis_config import load_config

# Global model instance
_MODEL_INSTANCE = None
_MODEL_CONFIG = None

class ModelManager:
    """Singleton manager for Qwen2.5-VL model."""
    
    def __new__(cls):
        global _MODEL_INSTANCE
        if _MODEL_INSTANCE is None:
            _MODEL_INSTANCE = super().__new__(cls)
            _MODEL_INSTANCE._initialized = False
        return _MODEL_INSTANCE
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration."""
        global _MODEL_CONFIG
        if _MODEL_CONFIG is None:
            _MODEL_CONFIG = load_config()
        self.config = _MODEL_CONFIG
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the Qwen2.5-VL model.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.model is not None and not force_reload:
            print("✅ Model already loaded, reusing existing instance")
            return True
        
        try:
            # Check if PyTorch is available
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from transformers import AutoProcessor
                TORCH_AVAILABLE = True
            except ImportError:
                print("⚠️  PyTorch not available, using mock mode")
                TORCH_AVAILABLE = False
                return False
            
            if not TORCH_AVAILABLE:
                return False
            
            print("🔄 Loading Qwen2.5-VL model...")
            model_name = self.config['model']['name']
            processor_name = self.config['model']['processor_name']
            print(f"📁 Model: {model_name}")
            print(f"📁 Processor: {processor_name}")
            
            # Load model components
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                processor_name,
                trust_remote_code=True
            )
            
            # Use the correct model class for Qwen2.5-VL
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            if not self.load_model():
                return None
        return self.model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        if self.tokenizer is None:
            if not self.load_model():
                return None
        return self.tokenizer
    
    def get_processor(self):
        """Get the loaded processor."""
        if self.processor is None:
            if not self.load_model():
                return None
        return self.processor
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_loaded": self.is_model_loaded(),
            "model_name": self.config['model']['name'] if self.config else None,
            "processor_name": self.config['model']['processor_name'] if self.config else None,
            "device": str(self.model.device) if self.model else None,
            "dtype": str(self.model.dtype) if self.model else None
        }
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.processor
            self.model = None
            self.tokenizer = None
            self.processor = None
            print("🗑️  Model unloaded from memory")

# Global instance
model_manager = ModelManager() 