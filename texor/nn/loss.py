from typing import Union, Optional
import numpy as np
from ..core import Tensor
from ..core.backend import backend

def binary_cross_entropy(input: Tensor, target: Tensor,
                        weight: Optional[np.ndarray] = None,
                        reduction: str = 'mean') -> Tensor:
    """Functional interface for binary cross entropy loss"""
    loss = BCELoss(weight=weight, reduction=reduction)
    return loss(input, target)

class Loss:
    """Base class for all loss functions"""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(predictions, targets)
        return self.forward(predictions, targets)
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
        
    def _validate_inputs(self, predictions: Tensor, targets: Union[Tensor, np.ndarray]) -> None:
        """Validate input shapes and types"""
        if not isinstance(predictions, Tensor):
            raise TypeError("predictions must be a Tensor")
            
        if not isinstance(targets, (Tensor, np.ndarray)):
            raise TypeError("targets must be a Tensor or numpy array")
            
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

class MSELoss(Loss):
    """Mean Squared Error Loss"""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.mse(predictions, targets)
        diff = predictions - targets
        return Tensor(np.mean(diff.numpy() ** 2))
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return 2 * (predictions - targets) / np.prod(predictions.shape)

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss with built-in softmax"""
    
    def __init__(self, weight: Optional[np.ndarray] = None, 
                 ignore_index: int = -100,
                 reduction: str = 'mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.cross_entropy(predictions, targets, 
                                      self.weight, self.ignore_index)
        
        # Apply softmax
        pred = predictions.numpy()
        exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
        
        # Compute cross entropy
        target_dist = targets.numpy()
        losses = -np.sum(target_dist * np.log(softmax_pred + 1e-7), axis=-1)
        
        # Apply weight if provided
        if self.weight is not None:
            losses = losses * self.weight
            
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = np.any(target_dist == self.ignore_index, axis=-1)
            losses = np.where(mask, 0, losses)
            
        # Reduction
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Gradient of cross entropy with respect to logits
        softmax = predictions.softmax()
        return softmax - targets

class BCELoss(Loss):
    """Binary Cross Entropy Loss"""
    
    def __init__(self, weight: Optional[np.ndarray] = None,
                 reduction: str = 'mean'):
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.binary_cross_entropy(predictions, targets, self.weight)
            
        pred = predictions.numpy()
        target = targets.numpy()
        
        # Clip predictions for numerical stability
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        losses = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        
        if self.weight is not None:
            losses = losses * self.weight
            
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pred = predictions.numpy()
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return (pred - targets) / (pred * (1 - pred))

class L1Loss(Loss):
    """Mean Absolute Error Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.l1_loss(predictions, targets)
            
        losses = np.abs(predictions.numpy() - targets.numpy())
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return np.sign(predictions - targets)

class HuberLoss(Loss):
    """Huber Loss"""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if backend.current == 'tensorflow':
            return backend.huber_loss(predictions, targets, self.delta)
            
        diff = predictions.numpy() - targets.numpy()
        abs_diff = np.abs(diff)
        
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        losses = 0.5 * quadratic ** 2 + self.delta * linear
        
        if self.reduction == 'mean':
            return Tensor(np.mean(losses))
        elif self.reduction == 'sum':
            return Tensor(np.sum(losses))
        return Tensor(losses)
        
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        abs_diff = np.abs(diff.numpy())
        return np.where(abs_diff <= self.delta, 
                       diff,
                       self.delta * np.sign(diff))

__all__ = [
    # Classes
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    'HuberLoss',
    
    # Functions
    'binary_cross_entropy',
    'get_loss_function'
]

# Factory function
def get_loss_function(name: str) -> Loss:
    """Get loss function by name"""
    losses = {
        'mse': MSELoss,
        'crossentropy': CrossEntropyLoss,
        'categorical_crossentropy': CrossEntropyLoss,
        'bce': BCELoss,
        'binary_crossentropy': BCELoss,
        'l1': L1Loss,
        'mae': L1Loss,
        'huber': HuberLoss
    }
    
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")
        
    return losses[name]()