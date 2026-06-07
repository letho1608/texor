from typing import Optional, Any, List
import numpy as np
from ...core.native_tensor import Tensor

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
        
    def parameters(self) -> List[Tensor]:
        """Get all parameters of the layer"""
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Layer):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Layer):
                        params.extend(item.parameters())
        return params
        
    def state_dict(self) -> dict:
        """Get layer state"""
        state = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                state[name] = value
            elif isinstance(value, Layer):
                state[name] = value.state_dict()
            elif isinstance(value, (list, tuple)):
                state[name] = []
                for item in value:
                    if isinstance(item, Tensor):
                        state[name].append(item)
                    elif isinstance(item, Layer):
                        state[name].append(item.state_dict())
                    else:
                        state[name].append(None) # Keep indexing
        return state
        
    def save(self, path: str) -> None:
        """Save layer state to file"""
        import pickle
        state = self.state_dict()
        # Convert Tensors to numpy for pickling safely
        def to_numpy(obj):
            if isinstance(obj, Tensor):
                return obj.data
            elif isinstance(obj, dict):
                return {k: to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_numpy(v) for v in obj]
            return obj
            
        with open(path, 'wb') as f:
            pickle.dump(to_numpy(state), f)
            
    def load(self, path: str) -> None:
        """Load layer state from file"""
        import pickle
        with open(path, 'rb') as f:
            state_data = pickle.load(f)
            
        # Convert numpy back to Tensors (or just assign data)
        def from_numpy(obj, data):
            if isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(obj, k):
                        attr = getattr(obj, k)
                        from_numpy(attr, v)
            elif isinstance(data, list):
                if isinstance(obj, (list, tuple)):
                    for i, v in enumerate(data):
                        if i < len(obj):
                            from_numpy(obj[i], v)
            elif isinstance(data, np.ndarray):
                if isinstance(obj, Tensor):
                    obj.data = data
                
        from_numpy(self, state_data)
