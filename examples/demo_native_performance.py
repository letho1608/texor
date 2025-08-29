#!/usr/bin/env python3
"""
Demo script showcasing Texor native performance and capabilities
"""

import texor
import numpy as np
import time
from texor.nn.layers import Linear, Sequential
from texor.nn.activations import ReLU
from texor.optim.optimizers import Adam
from texor.nn.loss import MSELoss

def main():
    print("🚀 TEXOR NATIVE PERFORMANCE DEMO")
    print("=" * 50)
    
    # 1. Basic Tensor Operations
    print("\n📊 1. BASIC TENSOR OPERATIONS")
    print("-" * 30)
    
    # Create tensors
    a = texor.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = texor.tensor([[7, 8, 9], [10, 11, 12]], requires_grad=True)
    
    print(f"Tensor A: {a}")
    print(f"Tensor B: {b}")
    
    # Operations
    c = a + b
    d = a * b
    e = a @ b.T
    
    print(f"A + B: {c}")
    print(f"A * B: {d}")
    print(f"A @ B.T: {e}")
    
    # 2. Gradient Computation
    print("\n🧠 2. AUTOMATIC DIFFERENTIATION")
    print("-" * 30)
    
    x = texor.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = texor.tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    # Complex computation
    z = (x * y).sum()
    print(f"z = (x * y).sum() = {z}")
    
    # Backpropagation
    z.backward()
    print(f"dz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")
    
    # 3. Neural Network
    print("\n🤖 3. NEURAL NETWORK TRAINING")
    print("-" * 30)
      # Create a simple model
    model = Sequential(
        Linear(3, 10),
        ReLU(),
        Linear(10, 1)
    )
    
    # Generate synthetic data
    X = texor.randn((100, 3))
    y = texor.randn((100, 1))
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
    
    print("Training neural network...")
    losses = []
    
    for epoch in range(10):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    # 4. Performance Benchmark
    print("\n⚡ 4. PERFORMANCE BENCHMARK")
    print("-" * 30)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Matrix multiplication benchmark
        a = texor.randn((size, size))
        b = texor.randn((size, size))
        
        start_time = time.time()
        c = a @ b
        end_time = time.time()
        
        print(f"  Matrix multiplication: {(end_time - start_time)*1000:.2f}ms")
        
        # Element-wise operations
        start_time = time.time()
        d = a * b + a - b
        end_time = time.time()
        
        print(f"  Element-wise ops: {(end_time - start_time)*1000:.2f}ms")
        
        # Activation functions
        start_time = time.time()
        e = c.relu()
        end_time = time.time()
        
        print(f"  ReLU activation: {(end_time - start_time)*1000:.2f}ms")
    
    # 5. Device Information
    print("\n💻 5. DEVICE INFORMATION")
    print("-" * 30)
    
    print(f"Current device: {texor.get_device()}")
    print(f"CUDA available: {texor.cuda_is_available()}")
    
    # 6. Advanced Features
    print("\n🔬 6. ADVANCED FEATURES")
    print("-" * 30)
    
    # Gradient checkpointing example
    x = texor.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    # Complex computation with multiple operations
    y = x.pow(2).sum().sqrt()
    print(f"y = sqrt(sum(x^2)) = {y}")
    
    y.backward()
    print(f"dy/dx = {x.grad}")
    
    # Tensor manipulation
    matrix = texor.randn((4, 4))
    print(f"\nOriginal matrix shape: {matrix.shape}")
    
    reshaped = matrix.reshape((2, 8))
    print(f"Reshaped: {reshaped.shape}")
    
    transposed = matrix.T
    print(f"Transposed: {transposed.shape}")
    
    print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
    print("Texor Native Framework is ready for production use! 🎉")

if __name__ == "__main__":
    main()
