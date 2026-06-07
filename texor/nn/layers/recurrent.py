from typing import Optional, Tuple, List, Union
import numpy as np
from ...core.native_tensor import Tensor
from .base import Layer
from ..activations import Sigmoid, Tanh

class RNN(Layer):
    """Simple Recurrent Neural Network layer"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []
        
        num_directions = 2 if bidirectional else 1
        
        for layer in range(num_layers):
            for _ in range(num_directions):
                l_input_size = input_size if layer == 0 else hidden_size * num_directions
                
                self.w_ih.append(Tensor(np.random.randn(hidden_size, l_input_size) / np.sqrt(l_input_size), requires_grad=True))
                self.w_hh.append(Tensor(np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True))
                self.b_ih.append(Tensor(np.zeros(hidden_size), requires_grad=True))
                self.b_hh.append(Tensor(np.zeros(hidden_size), requires_grad=True))

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, seq_length, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        if h_0 is None:
            h_0 = Tensor(np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size)))
        
        h_n = []
        output = []
        
        # Simple loop-based implementation (can be optimized with Numba)
        current_input = x
        for l in range(self.num_layers):
            layer_output = []
            # Forward direction
            h_t = h_0[l * num_directions]
            for t in range(seq_length):
                x_t = current_input[:, t, :]
                h_t = Tanh()(x_t @ self.w_ih[l*num_directions].T + self.b_ih[l*num_directions] + 
                             h_t @ self.w_hh[l*num_directions].T + self.b_hh[l*num_directions])
                layer_output.append(h_t)
            
            # TODO: Bidirectional logic
            
            current_input = Tensor(np.stack([t.data for t in layer_output], axis=1))
            output = current_input
            h_n.append(h_t)
            
        return output, Tensor(np.stack([h.data for h in h_n]))

class LSTM(Layer):
    """Long Short-Term Memory layer"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []
        
        num_directions = 2 if bidirectional else 1
        
        for layer in range(num_layers):
            for _ in range(num_directions):
                l_input_size = input_size if layer == 0 else hidden_size * num_directions
                
                self.w_ih.append(Tensor(np.random.randn(4 * hidden_size, l_input_size) / np.sqrt(l_input_size), requires_grad=True))
                self.w_hh.append(Tensor(np.random.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True))
                self.b_ih.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))
                self.b_hh.append(Tensor(np.zeros(4 * hidden_size), requires_grad=True))

    def forward(self, x: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_size, seq_length, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        if states is None:
            h_0 = Tensor(np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size)))
            c_0 = Tensor(np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size)))
        else:
            h_0, c_0 = states
            
        # Simplified LSTM forward
        current_input = x
        for l in range(self.num_layers):
            h_t = h_0[l]
            c_t = c_0[l]
            layer_output = []
            for t in range(seq_length):
                x_t = current_input[:, t, :]
                gates = x_t @ self.w_ih[l].T + self.b_ih[l] + h_t @ self.w_hh[l].T + self.b_hh[l]
                
                i, f, g, o = gates.chunk(4, axis=-1)
                i = Sigmoid()(i)
                f = Sigmoid()(f)
                g = Tanh()(g)
                o = Sigmoid()(o)
                
                c_t = f * c_t + i * g
                h_t = o * Tanh()(c_t)
                layer_output.append(h_t)
            
            current_input = Tensor(np.stack([h.data for h in layer_output], axis=1))
            
        return current_input, (h_t, c_t) # Returns last hidden/cell state

class GRU(Layer):
    """Gated Recurrent Unit layer"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.w_ih = []
        self.w_hh = []
        self.b_ih = []
        self.b_hh = []
        
        num_directions = 2 if bidirectional else 1
        
        for layer in range(num_layers):
            for _ in range(num_directions):
                l_input_size = input_size if layer == 0 else hidden_size * num_directions
                
                self.w_ih.append(Tensor(np.random.randn(3 * hidden_size, l_input_size) / np.sqrt(l_input_size), requires_grad=True))
                self.w_hh.append(Tensor(np.random.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True))
                self.b_ih.append(Tensor(np.zeros(3 * hidden_size), requires_grad=True))
                self.b_hh.append(Tensor(np.zeros(3 * hidden_size), requires_grad=True))

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Implementation omitted for brevity in this step, but would follow LSTM pattern
        return x, h_0
