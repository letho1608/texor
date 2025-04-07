"""Transformer model implementations"""
from typing import Optional, Dict
import numpy as np
from ...core import Tensor
from ..layers import Layer, Linear, Sequential, Embedding, LayerNorm
from .. import functional as F
from ..advanced_layers import TransformerEncoderLayer

class TransformerEncoder(Layer):
    """Transformer encoder"""
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", norm: Optional[Layer] = None):
        super().__init__()
        self.layers = Sequential([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = norm if norm is not None else LayerNorm(d_model)
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class BERTEmbeddings(Layer):
    """BERT embeddings"""
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, hidden_size)
        self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = Embedding(2, hidden_size)
        
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = dropout
        
    def forward(self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None) -> Tensor:
        seq_length = input_ids.shape[1]
        
        position_ids = Tensor(np.arange(seq_length)[np.newaxis, :])
        
        if token_type_ids is None:
            token_type_ids = F.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        
        if self.dropout > 0:
            embeddings = F.dropout(embeddings, p=self.dropout)
            
        return embeddings

class BERT(Layer):
    """BERT model implementation"""
    def __init__(self, vocab_size: int, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12,
                 intermediate_size: int = 3072, dropout: float = 0.1,
                 max_position_embeddings: int = 512):
        super().__init__()
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        self.encoder = TransformerEncoder(
            d_model=hidden_size,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=intermediate_size,
            dropout=dropout
        )
        
        self.pooler = Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None,
                token_type_ids: Optional[Tensor] = None) -> Dict[str, Tensor]:
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            mask=attention_mask
        )
        
        pooled_output = self.pooler(encoder_outputs[:, 0])
        
        return {
            'last_hidden_state': encoder_outputs,
            'pooler_output': pooled_output
        }

def bert_base_uncased(vocab_size: int) -> BERT:
    """BERT base model (uncased)"""
    return BERT(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072
    )

def bert_large_uncased(vocab_size: int) -> BERT:
    """BERT large model (uncased)"""
    return BERT(
        vocab_size=vocab_size,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096
    )