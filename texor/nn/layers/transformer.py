from typing import Optional, Tuple, Union
import numpy as np
from ...core.native_tensor import Tensor
from .base import Layer
from .linear import Linear
from .normalization import LayerNorm
from .utility import Dropout
from .attention import MultiheadAttention
from .. import functional as F

class TransformerEncoderLayer(Layer):
    """Transformer encoder layer"""

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
        # Self-attention block
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block
        if self.activation == 'relu':
            activation_fn = F.relu
        elif self.activation == 'gelu':
            activation_fn = F.gelu
        else:
            raise ValueError(f"Activation {self.activation} not supported")
            
        src2 = self.linear2(self.dropout(activation_fn(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderLayer(Layer):
    """Transformer decoder layer"""

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
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        if self.activation == 'relu':
            activation_fn = F.relu
        elif self.activation == 'gelu':
            activation_fn = F.gelu
        else:
            raise ValueError(f"Activation {self.activation} not supported")

        tgt2 = self.linear2(self.dropout(activation_fn(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerEncoder(Layer):
    """Transformer encoder"""
    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        from .container import ModuleList
        self.layers = ModuleList([
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
    """Transformer decoder"""
    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        from .container import ModuleList
        self.layers = ModuleList([
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
    """Full Transformer model"""
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
