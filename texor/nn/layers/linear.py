import numpy as np
from ...core.native_tensor import Tensor
from .base import Layer

class Linear(Layer):
    """Fully connected layer with optimized initialization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features        
        
        from .. import init
        self.weight = Tensor(np.empty((in_features, out_features)), requires_grad=True)
        init.kaiming_uniform(self.weight, nonlinearity='relu')
        
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None
            
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using optimized matrix multiplication"""
        output = inputs @ self.weight  # Use native tensor matmul
        if self.bias is not None:
            output = output + self.bias  # Use native tensor addition
        return output

class Embedding(Layer):
    """Embedding layer that maps indices to dense vectors"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        from .. import init
        self.weight = Tensor(np.empty((num_embeddings, embedding_dim)), requires_grad=True)
        init.xavier_uniform(self.weight)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using embedding lookup"""
        from ...core.native_backend import backend
        return backend.embedding(inputs, self.weight)
