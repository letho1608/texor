from typing import Optional, List, Any, Union
from ...core.native_tensor import Tensor
from .base import Layer

class Sequential(Layer):
    """Sequential container for layers"""
    
    def __init__(self, *layers: Layer):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)
        
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
        
    def train(self) -> None:
        super().train()
        for layer in self.layers:
            layer.train()
            
    def eval(self) -> None:
        super().eval()
        for layer in self.layers:
            layer.eval()

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]
        
    def __len__(self) -> int:
        return len(self.layers)

class ModuleList(Layer):
    """Holds submodules in a list"""
    def __init__(self, modules: Optional[List[Layer]] = None):
        super().__init__()
        self.modules = modules if modules is not None else []
        
    def __getitem__(self, idx: int) -> Layer:
        return self.modules[idx]
        
    def __len__(self) -> int:
        return len(self.modules)
        
    def append(self, module: Layer) -> None:
        self.modules.append(module)

    def extend(self, modules: List[Layer]) -> None:
        self.modules.extend(modules)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList should not be called directly")
