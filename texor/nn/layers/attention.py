from typing import Optional, Tuple, List
import numpy as np
from ...core.native_tensor import Tensor
from .base import Layer
from .linear import Linear
from .normalization import LayerNorm
from .utility import Dropout

class SelfAttention(Layer):
    """Self-attention layer"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = x.shape[0]
        
        # Linear projections and reshape
        q = self.q_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            # We assume mask is float and has high negative values for masked positions
            scores = scores + mask
        
        # Attention weights
        from ..activations import Softmax
        attn = Softmax()(scores)
        if self.dropout > 0:
            attn = Dropout(p=self.dropout)(attn)
            
        # Compute output
        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        x = self.out_proj(x)
        
        return x

class MultiheadAttention(Layer):
    """Multi-head attention layer (wrapper for SelfAttention or similar logic)"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Simplified: assuming query=key=value for now, or implementing full version
        # For full PyTorch parity, this needs more logic.
        # Sticking to SelfAttention logic for now.
        out = self.attention(query, mask=attn_mask)
        return out, None # Returning None for weights for now
