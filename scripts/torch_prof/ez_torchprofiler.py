import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        with record_function("forward_fc1"):
            x = self.fc1(x)
        
        with record_function("forward_activation"):
            x = self.relu(x)
            x = self.dropout(x)
        
        with record_function("forward_fc2"):
            x = self.fc2(x)
            x = self.relu(x)
        
        with record_function("forward_fc3"):
            x = self.fc3(x)
        
        return x

# 2. Create dummy data
def create_dummy_data(batch_size=128, num_batches=100):
    """Create simulated MNIST-style data"""
    # Simulate input data (batch_size, 784) - similar to flattened 28x28 images
    x_data = torch.randn(num_batches * batch_size, 784)
    # Simulate label data (batch_size,) - 10 classes
    y_data = torch.randint(0, 10, (num_batches * batch_size,))
    
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 3. Training function with profiling
def train_with_profiling():
    # Initialize TensorBoard writer
    writer = SummaryWriter('logs/simple_model_experiment')
    
    # Create model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loader
    train_loader = create_dummy_data(batch_size=64, num_batches=50)
    
    # 4. Configure profiler
    prof = profile(
        activities=[
            ProfilerActivity.CPU, 
            ProfilerActivity.CUDA
        ] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        
        schedule=torch.profiler.schedule(
            wait=2,      # Skip first 2 steps
            warmup=2,    # Warmup for 2 steps (data not recorded)
            active=6,    # Analyze 6 steps
            repeat=1     # Repeat cycle only once
        ),
        
        on_trace_ready=tensorboard_trace_handler(
            'logs/profiler_traces'
        ),
        
        record_shapes=True,      # Record tensor shapes
        profile_memory=True,     # Analyze memory usage
        with_stack=True,         # Record call stack
        with_flops=True,         # Calculate FLOPs
        
        # Experimental features
        with_modules=True        # Record module info (requires PyTorch 1.8+)
    )
    
    # Start training
    model.train()
    total_steps = 0
    
    print("Starting training and profiling...")
    prof.start()
    
    for epoch in range(3):  # Run for 3 epochs
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass (mark key operations with record_function)
            with record_function("forward_pass"):
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass
            with record_function("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log to TensorBoard
            if total_steps % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), total_steps)
                writer.add_scalar('Accuracy/train', 100. * correct / total, total_steps)
                
                # Log weight distributions
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        writer.add_histogram(f'weights/{name}', param, total_steps)
                        writer.add_histogram(f'gradients/{name}', param.grad, total_steps)
            
            # Execute profiler step
            prof.step()
            total_steps += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Statistics at the end of each epoch
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch} completed - Average Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Accuracy: {epoch_acc:.2f}%')
    
    # Stop profiler and save results
    prof.stop()
    
    # 5. Print key performance metrics
    print("\n=== Profiling Results ===")
    
    # Print timing of key operations
    if prof is not None:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        # Save detailed performance report
        with open("performance_report.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed! You can use TensorBoard to view the results.")

# 6. Additional profiling analysis function
def detailed_profiling_analysis():
    """More detailed performance analysis example"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    data = torch.randn(64, 784).to(device)
    
    # Use profile for detailed analysis
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run several forward passes
        for _ in range(10):
            output = model(data)
            loss = output.sum()
            loss.backward()
    
    # Print results sorted by different criteria
    print("=== Sorted by CPU Time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    
    print("\n=== Sorted by CUDA Time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\n=== Sorted by Memory Usage ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=5))

# 7. Helper function to start TensorBoard
def start_tensorboard():
    """Instructions for starting TensorBoard"""
    print("\n" + "="*50)
    print("TensorBoard Startup Instructions:")
    print("="*50)
    print("Run the following command in your terminal to start TensorBoard:")
    print("tensorboard --logdir logs --port 8098")
    print("\nThen visit: http://localhost:8098")
    print("\nYou can view the following:")
    print("- Scalars: Training loss and accuracy curves")
    print("- Profiler: Performance analysis results (operator timing, memory usage, etc.)")
    print("- Graphs: Model computation graph")
    print("- Histograms: Weight and gradient distributions")
    print("="*50)

if __name__ == "__main__":
    # Run main training function
    train_with_profiling()
    
    # Run detailed analysis (optional)
    print("\nRunning detailed performance analysis...")
    detailed_profiling_analysis()
    
    # Display TensorBoard startup instructions
    start_tensorboard()