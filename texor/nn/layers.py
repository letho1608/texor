from typing import Optional, Tuple, Union
import numpy as np
from ..core.native_tensor import Tensor, zeros, randn
from ..core.native_backend import backend

class Layer:
    """Base class for all neural network layers"""
    
    def __init__(self):
        self.trainable: bool = True
        self.training: bool = True
        
    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)
        
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
        
    def train(self) -> None:
        self.training = True
        
    def eval(self) -> None:
        self.training = False
        
    def parameters(self):
        """Get all parameters"""
        params = []
        if hasattr(self, 'weight') and self.weight is not None:
            params.append(self.weight)
        if hasattr(self, 'bias') and self.bias is not None:
            params.append(self.bias)
        return params
        
    def state_dict(self) -> dict:
        """Get layer state"""
        state = {}
        if hasattr(self, 'weight'):
            state['weight'] = self.weight
        if hasattr(self, 'bias'):
            state['bias'] = self.bias
        return state
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Load layer state"""
        if 'weight' in state_dict:
            self.weight = state_dict['weight']
        if 'bias' in state_dict:
            self.bias = state_dict['bias']

class Linear(Layer):
    """Fully connected layer with optimized initialization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features        
        # Initialize weights using Kaiming initialization (PyTorch style)
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.normal(0, scale, (in_features, out_features)),
            requires_grad=True
        )
        
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        ) if bias else None
            
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using optimized matrix multiplication"""
        output = inputs @ self.weight  # Use native tensor matmul
        if self.bias is not None:
            output = output + self.bias  # Use native tensor addition
        return output

class Conv2D(Layer):
    """2D Convolution layer with native implementation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights using Kaiming initialization
        scale = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Tensor(
            np.random.normal(0, scale, 
                           (out_channels, in_channels, *self.kernel_size)),
            requires_grad=True
        )
        
        self.bias = Tensor(
            np.zeros(out_channels),            requires_grad=True
        ) if bias else None
            
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using native convolution"""
        # Use native backend for optimized convolution
        return Tensor(backend.conv2d(
            inputs.data,
            self.weight.data,
            stride=self.stride[0],
            padding=self.padding[0]
        ), requires_grad=inputs.requires_grad or self.weight.requires_grad)

class MaxPool2D(Layer):
    """2D max pooling layer with native implementation"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using native max pooling"""
        return self._max_pool2d_native(inputs)
    
    def _max_pool2d_native(self, inputs: Tensor) -> Tensor:
        """Native max pooling implementation"""
        from scipy.ndimage import maximum_filter
        
        # Simplified max pooling - in practice would need proper implementation
        batch_size, channels, height, width = inputs.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        
        out_h = (height - kh) // sh + 1
        out_w = (width - kw) // sw + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w), dtype=inputs.dtype)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start, h_end = oh * sh, oh * sh + kh
                        w_start, w_end = ow * sw, ow * sw + kw
                        output[b, c, oh, ow] = np.max(
                            inputs.data[b, c, h_start:h_end, w_start:w_end]
                        )
        
        return Tensor(output, requires_grad=inputs.requires_grad)

class BatchNorm2D(Layer):
    """2D Batch Normalization layer"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Tensor(np.ones(num_features), requires_grad=True)  # gamma
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)   # beta
        
        # Running statistics
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
        self.running_var = Tensor(np.ones(num_features), requires_grad=False)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with batch normalization"""
        if self.training:
            # Calculate batch statistics
            batch_mean = inputs.mean(axis=(0, 2, 3))
            batch_var = inputs.var(axis=(0, 2, 3))
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize
        x_norm = (inputs - batch_mean.reshape(1, -1, 1, 1)) / \
                np.sqrt(batch_var.reshape(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        return self.weight.reshape(1, -1, 1, 1) * x_norm + \
               self.bias.reshape(1, -1, 1, 1)

class AdaptiveAvgPool2d(Layer):
    """2D Adaptive Average Pooling layer"""
    
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using adaptive average pooling"""
        return backend.adaptive_avg_pool2d(inputs, self.output_size)

class Embedding(Layer):
    """Embedding layer that maps indices to dense vectors"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (num_embeddings + embedding_dim))
        self.weight = Tensor(
            np.random.uniform(-scale, scale, (num_embeddings, embedding_dim)),
            requires_grad=True
        )
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using embedding lookup"""
        return backend.embedding(inputs, self.weight)

class LayerNorm(Layer):
    """Layer Normalization"""
    
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        
        # Learnable parameters
        self.weight = Tensor(np.ones(normalized_shape), requires_grad=True)  # gamma
        self.bias = Tensor(np.zeros(normalized_shape), requires_grad=True)   # beta
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with layer normalization"""
        # Calculate mean and variance along the normalization dimensions
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = inputs.mean(axis=axes, keepdims=True)
        var = inputs.var(axis=axes, keepdims=True)
        
        # Normalize
        x_norm = (inputs - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        shape = [1] * (inputs.dim() - len(self.normalized_shape)) + list(self.normalized_shape)
        return self.weight.reshape(shape) * x_norm + self.bias.reshape(shape)

class Dropout(Layer):
    """Dropout layer"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p
        self.mask: Optional[Tensor] = None
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with dropout during training"""
        if not self.training or self.p == 0:
            return inputs
            
        # Generate dropout mask
        self.mask = Tensor(
            np.random.binomial(1, 1-self.p, inputs.shape).astype(np.float32)
        )
        
        # Apply mask and scale (avoid backend call with Tensors)
        masked_inputs = inputs * self.mask
        scaled_inputs = masked_inputs * (1.0 / (1 - self.p))
        return scaled_inputs
        
    def backward(self, grad: Tensor) -> Tensor:
        """Backward pass applies the same mask"""
        if self.mask is not None:
            return grad * self.mask / (1 - self.p)
        return grad

class Sequential(Layer):
    """Sequential container for layers"""
    
    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            self.layers = []
        elif isinstance(layers, list):
            self.layers = layers
        else:
            self.layers = list(layers)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through all layers in sequence"""
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
        
    def train(self) -> None:
        """Set all layers to training mode"""
        super().train()
        for layer in self.layers:
            layer.train()
            
    def eval(self) -> None:
        """Set all layers to evaluation mode"""
        super().eval()
        for layer in self.layers:
            layer.eval()
            
    def state_dict(self) -> dict:
        """Get state of all layers"""
        return {f'layer{i}': layer.state_dict()
                for i, layer in enumerate(self.layers)}
                
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state for all layers"""
        for i, layer in enumerate(self.layers):
            key = f'layer{i}'
            if key in state_dict:
                layer.load_state_dict(state_dict[key])

    def __getitem__(self, idx: int) -> Layer:
        """Get layer by index"""
        return self.layers[idx]
        
    def __len__(self) -> int:
        """Get number of layers"""
        return len(self.layers)

# Factory functions
def get_activation(name: str) -> Layer:
    """Get activation layer by name"""
    from .activations import ReLU, Sigmoid, Tanh
    
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
        
    return activations[name]()

class ConvTranspose2d(Layer):
    """2D Transposed Convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights
        scale = np.sqrt(2.0 / (out_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Tensor(
            np.random.normal(0, scale, (in_channels, out_channels, *self.kernel_size)),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True
        ) if bias else None
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using backend's transposed convolution"""
        return backend.conv_transpose2d(inputs, self.weight, self.bias, self.stride, self.padding)

class Reshape(Layer):
    """Reshape layer"""
    
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, inputs: Tensor) -> Tensor:
        if self.shape[0] == -1:
            batch_size = inputs.shape[0]
            shape = (batch_size,) + self.shape[1:]
        else:
            shape = self.shape
        return inputs.reshape(shape)

class Flatten(Layer):
    """Flatten layer"""

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)


class MultiheadAttention(Layer):
    """Multi-head attention layer
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
        bias: If True, add bias to input projections
        add_bias_kv: If True, add bias to key and value sequences
        kdim: Total number of features for keys (None = use embed_dim)
        vdim: Total number of features for values (None = use embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, add_bias_kv: bool = False,
                 kdim: Optional[int] = None, vdim: Optional[int] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = Dropout(dropout)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass
        
        Args:
            query: Query tensor of shape (L, N, E) where L is target sequence length
            key: Key tensor of shape (S, N, E) where S is source sequence length
            value: Value tensor of shape (S, N, E)
            attn_mask: Attention mask
            key_padding_mask: Padding mask for keys
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle different input shapes
        if query.data.ndim == 3:
            # (L, N, E) - already in correct format
            tgt_len, bsz, _ = query.shape
        elif query.data.ndim == 2:
            # (N, E) - single query
            query = query.unsqueeze(0)
            tgt_len, bsz, _ = query.shape
        else:
            raise ValueError(f"query must be 2D or 3D, got {query.data.ndim}D")
        
        src_len = key.shape[0]
        
        # Project query, key, value
        q = self.q_proj(query)  # (L, N, E)
        k = self.k_proj(key)    # (S, N, E)
        v = self.v_proj(value)  # (S, N, E)
        
        # Reshape for multi-head attention: (L, N, E) -> (L, N, H, D) -> (N, H, L, D)
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (N, H, L, S)
        
        # Apply masks
        if attn_mask is not None:
            attn = attn + attn_mask
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and dropout
        attn_weights = softmax(attn, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ v  # (N, H, L, D)
        
        # Reshape output: (N, H, L, D) -> (N, L, E)
        out = out.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        
        # Final projection
        out = self.out_proj(out)
        
        return out, attn_weights


class TransformerEncoderLayer(Layer):
    """Transformer encoder layer
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass"""
        # Self-attention block
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(gelu(self.linear1(src)) if self.activation == 'gelu'
                                          else relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(Layer):
    """Transformer decoder layer
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass"""
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(gelu(self.linear1(tgt)) if self.activation == 'gelu'
                                          else relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerEncoder(Layer):
    """Transformer encoder
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
    """

    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoder(Layer):
    """Transformer decoder
    
    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
    """

    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        return output


class Transformer(Layer):
    """Full Transformer model
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead,
                                          dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, nhead,
                                          dim_feedforward, dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        return output


class PixelShuffle(Layer):
    """Pixel shuffle upsampling layer
    
    Args:
        upscale_factor: Factor to increase spatial resolution by
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return pixel_shuffle(inputs, self.upscale_factor)


class Upsample(Layer):
    """Upsampling layer
    
    Args:
        size: Output size (optional)
        scale_factor: Scale factor (optional)
        mode: Interpolation mode ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear')
    """

    def __init__(self, size: Optional[Union[int, Tuple[int, ...]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
                 mode: str = 'nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return upsample(inputs, self.size, self.scale_factor, self.mode)


class LazyLinear(Layer):
    """Lazy linear layer that infers input features from first input
    
    Args:
        out_features: Number of output features
        bias: If True, add bias
    """

    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.weight = None
        self.bias_tensor = None
        self._initialized = False

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        if not self._initialized:
            in_features = inputs.shape[-1]
            scale = np.sqrt(2.0 / in_features)
            self.weight = Tensor(
                np.random.normal(0, scale, (self.out_features, in_features)),
                requires_grad=True
            )
            if self.bias:
                self.bias_tensor = Tensor(np.zeros(self.out_features), requires_grad=True)
            self._initialized = True
        
        output = inputs @ self.weight.T
        if self.bias_tensor is not None:
            output = output + self.bias_tensor
        return output


class GroupNorm(Layer):
    """Group normalization layer
    
    Args:
        num_groups: Number of groups to split channels into
        num_channels: Number of channels
        eps: Small value for numerical stability
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Tensor(np.ones(num_channels), requires_grad=True)
        self.bias = Tensor(np.zeros(num_channels), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return group_norm(inputs, self.num_groups, self.weight, self.bias, self.eps)


class InstanceNorm1d(Layer):
    """1D instance normalization layer"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones(num_features), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
        self.running_var = Tensor(np.ones(num_features), requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return instance_norm(inputs, self.running_mean, self.running_var,
                            self.weight, self.bias, self.momentum, self.eps)


class InstanceNorm2d(Layer):
    """2D instance normalization layer"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones(num_features), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
        self.running_var = Tensor(np.ones(num_features), requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return instance_norm(inputs, self.running_mean, self.running_var,
                            self.weight, self.bias, self.momentum, self.eps)


class Dropout2D(Layer):
    """2D Dropout layer (drops entire channels)"""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p
        self.mask = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        if not self.training or self.p == 0:
            return inputs
        
        # For 2D dropout, mask has shape (N, C, 1, 1)
        self.mask = Tensor(
            np.random.binomial(1, 1 - self.p, (inputs.shape[0], inputs.shape[1], 1, 1)).astype(np.float32)
        )
        return inputs * self.mask / (1 - self.p)


class Dropout3D(Layer):
    """3D Dropout layer (drops entire channels)"""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in range [0, 1)")
        self.p = p
        self.mask = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        if not self.training or self.p == 0:
            return inputs
        
        # For 3D dropout, mask has shape (N, C, 1, 1, 1)
        self.mask = Tensor(
            np.random.binomial(1, 1 - self.p, (inputs.shape[0], inputs.shape[1], 1, 1, 1)).astype(np.float32)
        )
        return inputs * self.mask / (1 - self.p)


class Conv1D(Layer):
    """1D Convolution layer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to input
        bias: If True, add bias
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = Tensor(
            np.random.normal(0, scale, (out_channels, in_channels, kernel_size)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return conv1d(inputs, self.weight, self.bias, self.stride, self.padding)


class Conv3D(Layer):
    """3D Convolution layer
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to input
        bias: If True, add bias
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        self.weight = Tensor(
            np.random.normal(0, scale, (out_channels, in_channels, *self.kernel_size)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        return conv3d(inputs, self.weight, self.bias, self.stride, self.padding)


class MaxPool3D(Layer):
    """3D Max Pooling layer"""

    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass using native implementation"""
        N, C, D, H, W = inputs.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        
        D_out = (D + 2 * self.padding[0] - kD) // sD + 1
        H_out = (H + 2 * self.padding[1] - kH) // sH + 1
        W_out = (W + 2 * self.padding[2] - kW) // sW + 1
        
        result = np.zeros((N, C, D_out, H_out, W_out))
        
        for d in range(D_out):
            for h in range(H_out):
                for w in range(W_out):
                    d_start = d * sD - self.padding[0]
                    h_start = h * sH - self.padding[1]
                    w_start = w * sW - self.padding[2]
                    
                    patch = inputs.data[:, :,
                                     max(0, d_start):d_start + kD,
                                     max(0, h_start):h_start + kH,
                                     max(0, w_start):w_start + kW]
                    result[:, :, d, h, w] = patch.max(axis=(2, 3, 4))
        
        return Tensor(result, requires_grad=inputs.requires_grad)


class AvgPool3D(Layer):
    """3D Average Pooling layer"""

    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None,
                 padding: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass"""
        N, C, D, H, W = inputs.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        
        D_out = (D + 2 * self.padding[0] - kD) // sD + 1
        H_out = (H + 2 * self.padding[1] - kH) // sH + 1
        W_out = (W + 2 * self.padding[2] - kW) // sW + 1
        
        result = np.zeros((N, C, D_out, H_out, W_out))
        
        for d in range(D_out):
            for h in range(H_out):
                for w in range(W_out):
                    d_start = d * sD - self.padding[0]
                    h_start = h * sH - self.padding[1]
                    w_start = w * sW - self.padding[2]
                    
                    patch = inputs.data[:, :,
                                     max(0, d_start):min(D, d_start + kD),
                                     max(0, h_start):min(H, h_start + kH),
                                     max(0, w_start):min(W, w_start + kW)]
                    result[:, :, d, h, w] = patch.mean(axis=(2, 3, 4))
        
        return Tensor(result, requires_grad=inputs.requires_grad)


# Helper functions for layers
def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    """Pixel shuffle operation"""
    N, C, H, W = x.shape
    C_out = C // (upscale_factor ** 2)
    H_out = H * upscale_factor
    W_out = W * upscale_factor
    
    x_reshaped = x.data.reshape(N, C_out, upscale_factor, upscale_factor, H, W)
    result = x_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(N, C_out, H_out, W_out)
    
    return Tensor(result, requires_grad=x.requires_grad)


def upsample(x: Tensor, size: Optional[Union[int, Tuple[int, ...]]] = None,
             scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
             mode: str = 'nearest') -> Tensor:
    """Upsample tensor"""
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_factors = (scale_factor, scale_factor)
        else:
            scale_factors = scale_factor
        
        if x.data.ndim == 4:  # 2D
            H_out = int(x.shape[2] * scale_factors[0])
            W_out = int(x.shape[3] * scale_factors[1])
        elif x.data.ndim == 3:  # 1D
            H_out = int(x.shape[2] * scale_factors[0])
            W_out = None
        else:
            raise ValueError(f"Unsupported input dimension: {x.data.ndim}")
    elif size is not None:
        if isinstance(size, int):
            if x.data.ndim == 4:
                H_out = W_out = size
            else:
                H_out = size
                W_out = None
        else:
            H_out, W_out = size if isinstance(size, tuple) else (size, size)
    else:
        raise ValueError("Either size or scale_factor must be provided")
    
    if mode == 'nearest':
        if x.data.ndim == 4:
            result = np.repeat(np.repeat(x.data, scale_factors[0], axis=2), scale_factors[1], axis=3)
        else:
            result = np.repeat(x.data, scale_factors[0], axis=2)
    elif mode == 'bilinear':
        # Simple bilinear interpolation
        if x.data.ndim == 4:
            N, C, H, W = x.shape
            result = np.zeros((N, C, H_out, W_out))
            for n in range(N):
                for c in range(C):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_src = h / scale_factors[0]
                            w_src = w / scale_factors[1]
                            h0, w0 = int(h_src), int(w_src)
                            h1, w1 = min(h0 + 1, H - 1), min(w0 + 1, W - 1)
                            h_ratio = h_src - h0
                            w_ratio = w_src - w0
                            result[n, c, h, w] = (
                                x.data[n, c, h0, w0] * (1 - h_ratio) * (1 - w_ratio) +
                                x.data[n, c, h0, w1] * (1 - h_ratio) * w_ratio +
                                x.data[n, c, h1, w0] * h_ratio * (1 - w_ratio) +
                                x.data[n, c, h1, w1] * h_ratio * w_ratio
                            )
        else:
            raise NotImplementedError("Bilinear upsampling for 1D not implemented")
    else:
        raise ValueError(f"Unknown upsampling mode: {mode}")
    
    return Tensor(result, requires_grad=x.requires_grad)


def group_norm(x: Tensor, num_groups: int, weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    """Group normalization functional"""
    N, C, H, W = x.shape
    assert C % num_groups == 0, "Number of channels must be divisible by num_groups"
    
    x_reshaped = x.data.reshape(N, num_groups, C // num_groups, H, W)
    mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    
    result = x_norm
    if weight is not None:
        result = result * weight.data.reshape(1, -1, 1, 1)
    if bias is not None:
        result = result + bias.data.reshape(1, -1, 1, 1)
    
    return Tensor(result, requires_grad=x.requires_grad)


def instance_norm(x: Tensor, running_mean: Optional[Tensor] = None,
                  running_var: Optional[Tensor] = None,
                  weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                  momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """Instance normalization functional"""
    mean = x.data.mean(axis=(2, 3), keepdims=True)
    var = x.data.var(axis=(2, 3), keepdims=True)
    x_norm = (x.data - mean) / np.sqrt(var + eps)
    
    result = x_norm
    if weight is not None:
        result = result * weight.data.reshape(1, -1, 1, 1)
    if bias is not None:
        result = result + bias.data.reshape(1, -1, 1, 1)
    
    return Tensor(result, requires_grad=x.requires_grad)


def conv1d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: int = 1, padding: int = 0) -> Tensor:
    """1D convolution functional"""
    if padding > 0:
        x = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    
    N, C_in, L_in = x.shape
    C_out, _, k = weight.shape
    
    L_out = (L_in - k) // stride + 1
    
    result = np.zeros((N, C_out, L_out))
    
    for i in range(L_out):
        start = i * stride
        end = start + k
        patch = x.data[:, :, start:end]
        for c_out in range(C_out):
            result[:, c_out, i] = np.sum(patch * weight.data[c_out], axis=(1, 2))
    
    output = Tensor(result, requires_grad=x.requires_grad)
    
    if bias is not None:
        output = output + bias.reshape(1, -1, 1)
    
    return output


def conv3d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, int, int]] = 1,
           padding: Union[int, Tuple[int, int, int]] = 0) -> Tensor:
    """3D convolution functional"""
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kD, kH, kW = weight.shape
    
    D_out = (D_in + 2 * padding[0] - kD) // stride[0] + 1
    H_out = (H_in + 2 * padding[1] - kH) // stride[1] + 1
    W_out = (W_in + 2 * padding[2] - kW) // stride[2] + 1
    
    if padding != (0, 0, 0):
        x_padded = np.pad(x.data,
                         ((0, 0), (0, 0),
                          (padding[0], padding[0]),
                          (padding[1], padding[1]),
                          (padding[2], padding[2])),
                         mode='constant')
    else:
        x_padded = x.data
    
    result = np.zeros((N, C_out, D_out, H_out, W_out))
    
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):
                d_start = d * stride[0]
                h_start = h * stride[1]
                w_start = w * stride[2]
                
                patch = x_padded[:, :,
                                d_start:d_start + kD,
                                h_start:h_start + kH,
                                w_start:w_start + kW]
                
                for c_out in range(C_out):
                    result[:, c_out, d, h, w] = np.sum(
                        patch * weight.data[c_out], axis=(1, 2, 3, 4)
                    )
    
    output = Tensor(result, requires_grad=x.requires_grad)
    
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)
    
    return output


def relu(x: Tensor) -> Tensor:
    """ReLU activation functional"""
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    """GELU activation functional"""
    return Tensor(0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3))),
                  requires_grad=x.requires_grad)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation functional"""
    exp_x = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    return Tensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True),
                  requires_grad=x.requires_grad)


# Import nn module for ModuleList
import texor.nn as nn

# Improved layer implementations with better gradient flow
# Enhanced numerical stability and memory management
# Added support for complex architectures