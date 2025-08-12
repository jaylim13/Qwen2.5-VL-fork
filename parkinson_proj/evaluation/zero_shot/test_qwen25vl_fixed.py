#!/usr/bin/env python3
"""
Quick test script for Qwen2.5-VL (Fixed version)
Tests image, video, and text inference capabilities
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

def test_model_loading():
    """Test if the model loads successfully"""
    print("🔄 Testing model loading...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("✅ Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def get_model_device(model):
    """Get the main device of the model for input placement"""
    # For multi-GPU models, we need to find the main device
    if hasattr(model, 'hf_device_map'):
        # Get the first device from the device map
        return list(model.hf_device_map.values())[0]
    else:
        # Fallback to the device of the first parameter
        return next(model.parameters()).device

def test_image_inference(model, processor):
    """Test image understanding"""
    print("\n🖼️  Testing image inference...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    
    try:
        start_time = time.time()
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # Move inputs to the correct device
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        print(f"✅ Image inference completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
    except Exception as e:
        print(f"❌ Image inference failed: {e}")

def test_text_only_inference(model, processor):
    """Test text-only conversation"""
    print("\n💬 Testing text-only inference...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the key features of Qwen2.5-VL?"},
    ]
    
    try:
        start_time = time.time()
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )
        # Move inputs to the correct device
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        print(f"✅ Text inference completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
    except Exception as e:
        print(f"❌ Text inference failed: {e}")

def test_multi_image_inference(model, processor):
    """Test multi-image understanding"""
    print("\n🖼️🖼️  Testing multi-image inference...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Compare these two images. Are they the same?"},
            ],
        }
    ]
    
    try:
        start_time = time.time()
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # Move inputs to the correct device
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        print(f"✅ Multi-image inference completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
    except Exception as e:
        print(f"❌ Multi-image inference failed: {e}")

def test_video_inference(model, processor):
    """Test video understanding"""
    print("\n🎥 Testing video inference...")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": "Describe what happens in this video."},
            ],
        }
    ]
    
    try:
        start_time = time.time()
        
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
        device = get_model_device(model)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        print(f"✅ Video inference completed in {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {output_text[0]}")
        
    except Exception as e:
        print(f"❌ Video inference failed: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Qwen2.5-VL Test Suite (Fixed Version)")
    print("=" * 50)
    
    # Test model loading
    model, processor = test_model_loading()
    if model is None or processor is None:
        print("❌ Cannot proceed without model loading")
        return
    
    # Test different capabilities
    test_text_only_inference(model, processor)
    test_image_inference(model, processor)
    test_multi_image_inference(model, processor)
    test_video_inference(model, processor)
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("💡 You can now use Qwen2.5-VL for your projects!")

if __name__ == "__main__":
    main() 