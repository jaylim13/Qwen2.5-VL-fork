#!/usr/bin/env python3
"""
Test script to verify that the QwenSFTTrainer fix works properly
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from src.trainer import QwenSFTTrainer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, input_ids=None, labels=None, **kwargs):
        # Simulate a simple forward pass
        if input_ids is not None:
            outputs = self.linear(input_ids.float())
            if labels is not None:
                loss = nn.functional.mse_loss(outputs.squeeze(), labels.float())
                return type('obj', (object,), {'loss': loss})()
            return type('obj', (object,), {'logits': outputs})()
        return type('obj', (object,), {'logits': torch.randn(1, 1)})()
    
    def eval(self):
        return self

def test_trainer():
    print("🧪 Testing QwenSFTTrainer fix...")
    
    # Create dummy model and data
    model = DummyModel()
    
    # Create dummy dataset
    class DummyDataset:
        def __init__(self, size=10):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 100, (10,)),
                'labels': torch.randn(1),
                'attention_mask': torch.ones(10)
            }
    
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset(5)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        logging_steps=1,
        report_to=None,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = QwenSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Test evaluation step
    print("📊 Testing evaluation step...")
    try:
        # Create dummy inputs
        inputs = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randn(2, 1),
            'attention_mask': torch.ones(2, 10)
        }
        
        # Test evaluation step
        eval_results = trainer.evaluation_step(model, inputs)
        print(f"✅ Evaluation step works: {eval_results}")
        
        # Test compute_metrics
        eval_preds = type('obj', (object,), {
            'predictions': [torch.tensor(0.5)],
            'label_ids': [torch.tensor(0.5)]
        })()
        
        metrics = trainer.compute_metrics(eval_preds)
        print(f"✅ Compute metrics works: {metrics}")
        
        print("🎉 All tests passed! The trainer fix should work.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_trainer() 