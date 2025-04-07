from typing import Optional, Tuple, List, Callable
import numpy as np
from ..core import Tensor
from ..core.backend import backend
from .layers import Layer, Conv2d as Conv2D, BatchNorm2d as BatchNorm2D, Sequential, Linear, LayerNorm, Dropout
from .activations import ReLU, Sigmoid, Tanh, Softmax, get_activation

class ResidualBlock(Layer):
    """Residual block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Skip connection
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential([
                Conv2D(in_channels, out_channels, kernel_size=1, stride=stride),
                BatchNorm2D(out_channels)
            ])
            
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = ReLU()(out)  # Using ReLU class instance
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
            
        out += identity
        out = ReLU()(out)
        
        return out

class LSTM(Layer):
    """Long Short-Term Memory layer
    
    Args:
        input_size: Number of expected features in the input x
        hidden_size: Number of features in the hidden state h
        num_layers: Number of recurrent layers (default: 1)
        dropout: Dropout probability for the output (default: 0.0)
        bidirectional: If True, becomes a bidirectional LSTM (default: False)
        
    Shapes:
        - Input: (batch_size, seq_length, input_size)
        - Output: (batch_size, seq_length, hidden_size * (2 if bidirectional else 1))
        - h_0: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
        - c_0: (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Initialize weights
        # Initialize weights and biases with proper typing
        self.w_ih: List[Tensor] = []  # Input-hidden weights
        self.w_hh: List[Tensor] = []  # Hidden-hidden weights
        self.b_ih: List[Tensor] = []  # Input-hidden biases
        self.b_hh: List[Tensor] = []  # Hidden-hidden biases
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            self.w_ih.append(Tensor(
                np.random.randn(4 * hidden_size, layer_input_size) / np.sqrt(layer_input_size),
                requires_grad=True
            ))
            self.w_hh.append(Tensor(
                np.random.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size),
                requires_grad=True
            ))
            self.b_ih.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))
            self.b_hh.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))
            
            if bidirectional:
                self.w_ih.append(Tensor(
                    np.random.randn(4 * hidden_size, layer_input_size) / np.sqrt(layer_input_size),
                    requires_grad=True
                ))
                self.w_hh.append(Tensor(
                    np.random.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size),
                    requires_grad=True
                ))
                self.b_ih.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))
                self.b_hh.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))
    
    def forward(self, x: Tensor,
                initial_states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x = backend.to_tensor(x)
        """Forward pass of LSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            initial_states: Optional tuple of (h_0, c_0) initial states
            
        Returns:
            tuple: (output, (h_n, c_n)) where:
                - output: Tensor of shape (batch_size, seq_length, hidden_size * (2 if bidirectional else 1))
                - h_n: Final hidden state for each layer
                - c_n: Final cell state for each layer
                
        Note:
            h_n and c_n have shape (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        if initial_states is None:
            h_0 = Tensor(np.zeros((self.num_layers * (2 if self.bidirectional else 1),
                                 batch_size, self.hidden_size)))
            c_0 = Tensor(np.zeros_like(h_0.numpy()))
            initial_states = (h_0, c_0)
            
        h_n = initial_states[0]
        c_n = initial_states[1]
        
        output = []
        for t in range(seq_length):
            h_t = []
            c_t = []
            x_t = x[:, t]
            
            for layer in range(self.num_layers):
                if layer > 0:
                    x_t = h_t[-1]
                
                # Forward direction
                h_forward = h_n[layer * (2 if self.bidirectional else 1)]
                c_forward = c_n[layer * (2 if self.bidirectional else 1)]
                
                gates = (x_t @ self.w_ih[layer].transpose() + self.b_ih[layer] +
                        h_forward @ self.w_hh[layer].transpose() + self.b_hh[layer])
                
                i, f, g, o = gates.chunk(4, axis=-1)
                i = Sigmoid()(i)
                f = Sigmoid()(f)
                g = Tanh()(g)
                o = Sigmoid()(o)
                
                c_forward = f * c_forward + i * g
                h_forward = o * Tanh()(c_forward)
                
                if self.bidirectional:
                    # Backward direction
                    h_backward = h_n[layer * 2 + 1]
                    c_backward = c_n[layer * 2 + 1]
                    
                    gates = (x_t @ self.w_ih[layer * 2 + 1].transpose() + self.b_ih[layer * 2 + 1] +
                            h_backward @ self.w_hh[layer * 2 + 1].transpose() + self.b_hh[layer * 2 + 1])
                    
                    i, f, g, o = gates.chunk(4, axis=-1)
                    i = Sigmoid()(i)
                    f = Sigmoid()(f)
                    g = Tanh()(g)
                    o = Sigmoid()(o)
                    
                    c_backward = f * c_backward + i * g
                    h_backward = o * Tanh()(c_backward)
                    
                    h_t.extend([h_forward, h_backward])
                    c_t.extend([c_forward, c_backward])
                else:
                    h_t.append(h_forward)
                    c_t.append(c_forward)
            
            output.append(Tensor(np.concatenate([t.numpy() for t in (h_t[-2:] if self.bidirectional else [h_t[-1]])], axis=-1)))
            h_n = Tensor(np.stack([t.numpy() for t in h_t]))
            c_n = Tensor(np.stack([t.numpy() for t in c_t]))
        
        output = Tensor(np.stack([t.numpy() for t in output], axis=1))
        return output, (h_n, c_n)

class SelfAttention(Layer):
    """Self-attention layer that allows the model to attend to different parts of the input

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads (default: 8)
        dropout: Dropout probability on attention weights (default: 0.0)
        
    Shapes:
        - Input: (batch_size, seq_length, embed_dim)
        - Output: (batch_size, seq_length, embed_dim)
        - Mask: (batch_size, seq_length, seq_length) or None
        
    Note:
        embed_dim must be divisible by num_heads
    """
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
        x = backend.to_tensor(x)
        if mask is not None:
            mask = backend.to_tensor(mask.astype(np.float32))
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
            scores = scores * mask + (1 - mask) * float('-inf')
        
        # Attention weights
        attn = Softmax()(scores)
        if self.dropout > 0:
            attn = Dropout(p=self.dropout)(attn)
            
        # Compute output
        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        x = self.out_proj(x)
        
        return x

class TransformerEncoderLayer(Layer):
    """Transformer encoder layer"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = SelfAttention(d_model, nhead, dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = dropout
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.activation = get_activation(activation)
        
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = backend.to_tensor(src)
        if src_mask is not None:
            src_mask = backend.to_tensor(src_mask.astype(np.float32))
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask)
        x = x + self._ff_block(self.norm2(x))
        return x
        
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, mask=attn_mask)
        return Dropout(p=self.dropout)(x) if self.dropout > 0 else x
        
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = Dropout(p=self.dropout)(x) if self.dropout > 0 else x
        x = self.linear2(x)
        return Dropout(p=self.dropout)(x) if self.dropout > 0 else x