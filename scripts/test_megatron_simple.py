import torch
import torch.nn as nn
import os
import sys

sys.path.append('/app')

def test_megatron_simple():
    print("Starting Megatron model test...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return self.layer_norm(x)
    
    device = "cuda"
    model = SimpleModel().to(device)
    input_data = torch.randn(4, 2048, 4096, device=device)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(input_data)
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.4f}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_megatron_simple()
