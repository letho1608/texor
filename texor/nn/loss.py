from typing import Union, Optional
import numpy as np
from ..core.native_tensor import Tensor

def binary_cross_entropy(input: Tensor, target: Tensor,
                        weight: Optional[np.ndarray] = None,
                        reduction: str = 'mean') -> Tensor:
    """Functional interface for binary cross entropy loss"""
    loss = BCELoss(weight=weight, reduction=reduction)
    return loss(input, target)

def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """Functional interface for mean squared error loss"""
    loss = MSELoss(reduction=reduction)
    return loss(input, target)

def cross_entropy(input: Tensor, target: Tensor, weight: Optional[np.ndarray] = None,
                 ignore_index: int = -100, reduction: str = 'mean') -> Tensor:
    """Functional interface for cross entropy loss"""
    loss = CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss(input, target)

class Loss:
    """Base class for all loss functions"""
    
    def __init__(self, reduction: str = 'mean'):
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction}")
        self.reduction = reduction
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(predictions, targets)
        return self.forward(predictions, targets)
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
        
    def _validate_inputs(self, predictions: Tensor, targets: Union[Tensor, np.ndarray]) -> None:
        """Validate input shapes and types"""
        if not isinstance(predictions, Tensor):
            raise TypeError("predictions must be a Tensor")
            
        if not isinstance(targets, (Tensor, np.ndarray)):
            raise TypeError("targets must be a Tensor or numpy array")
            
    def _apply_reduction(self, loss: Tensor) -> Tensor:
        """Apply reduction to loss values"""
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class MSELoss(Loss):
    """Mean Squared Error Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        # Calculate squared differences
        diff = predictions - targets
        squared_diff = diff * diff
        
        # Apply reduction while preserving gradients
        return self._apply_reduction(squared_diff)

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss with built-in softmax"""
    
    def __init__(self, weight: Optional[np.ndarray] = None, 
                 ignore_index: int = -100,
                 reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        # Check if targets are class indices or one-hot
        if targets.data.ndim == predictions.data.ndim - 1:
            # Class indices - convert to one-hot for easier computation
            num_classes = predictions.shape[-1]
            target_data = targets.data
            target_one_hot = np.zeros(predictions.shape)
            target_indices = target_data.flatten().astype(int)
            batch_indices = np.arange(target_indices.size)
            target_one_hot.reshape(-1, num_classes)[batch_indices, target_indices] = 1
            target_one_hot = target_one_hot.reshape(predictions.shape)
            targets = Tensor(target_one_hot)
        
        # Apply softmax and compute log
        softmax_pred = predictions.softmax(axis=-1)
        
        # Compute cross entropy: -sum(target * log(pred))
        eps = 1e-7
        losses = -(targets * (softmax_pred + eps).log()).sum(axis=-1)
        
        # Apply weight if provided
        if self.weight is not None:
            weight_tensor = Tensor(self.weight)
            if targets.data.ndim > 1:
                class_indices = np.argmax(targets.data, axis=-1)
                weight_mask = weight_tensor.data[class_indices]
                losses = losses * Tensor(weight_mask)
            else:
                losses = losses * weight_tensor
            
        # Handle ignore_index
        if self.ignore_index >= 0:
            if targets.data.ndim > 1:
                mask = np.any(targets.data == self.ignore_index, axis=-1)
            else:
                mask = targets.data == self.ignore_index
            losses = losses * Tensor((~mask).astype(np.float32))
        
        return self._apply_reduction(losses)

class BCELoss(Loss):
    """Binary Cross Entropy Loss"""
    
    def __init__(self, weight: Optional[np.ndarray] = None,
                 reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        # Stay in Tensor land to preserve gradients
        eps = 1e-7
        losses = -(targets * (predictions + eps).log() + (1 - targets) * (1 - predictions + eps).log())
        
        if self.weight is not None:
            losses = losses * Tensor(self.weight)
        
        return self._apply_reduction(losses)

class L1Loss(Loss):
    """Mean Absolute Error Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        # Calculate absolute differences using tensor operations
        diff = predictions - targets
        return self._apply_reduction(diff.abs())

class HuberLoss(Loss):
    """Huber Loss (smooth L1 loss)"""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        diff = predictions - targets
        abs_diff = diff.abs()
        
        # Piecewise function implemented with tensor ops to preserve graph
        # quadratic if abs_diff <= delta, else linear
        # min(abs_diff, delta) = abs_diff - relu(abs_diff - delta)
        quadratic_part = abs_diff - (abs_diff - self.delta).relu()
        linear_part = abs_diff - quadratic_part
        
        losses = 0.5 * quadratic_part * quadratic_part + linear_part * self.delta
        
        return self._apply_reduction(losses)

class SmoothL1Loss(Loss):
    """Smooth L1 Loss (similar to Huber with beta parameter)"""
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        diff = predictions - targets
        abs_diff = diff.abs()
        
        # Piecewise function: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
        # Piece in [0, beta]: quadratic_part = min(abs_diff, beta)
        quadratic_part = abs_diff - (abs_diff - self.beta).relu()
        # Piece in [beta, inf]: linear_part = relu(abs_diff - beta)
        linear_part = (abs_diff - self.beta).relu()
        
        losses = 0.5 * quadratic_part * quadratic_part / self.beta + linear_part
        
        return self._apply_reduction(losses)

class KLDivLoss(Loss):
    """Kullback-Leibler Divergence Loss"""
    
    def __init__(self, reduction: str = 'mean', log_target: bool = False):
        super().__init__(reduction)
        self.log_target = log_target
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        if self.log_target:
            # Both inputs are in log space
            # losses = exp(target) * (target - predictions)
            losses = targets.exp() * (targets - predictions)
        else:
            # Predictions in log space, targets in probability space
            # losses = target * (log(target) - predictions)
            eps = 1e-7
            losses = targets * ((targets + eps).log() - predictions)
            
        losses = losses.sum(axis=-1)
        return self._apply_reduction(losses)

def get_loss_function(name: str, **kwargs) -> Loss:
    """Factory function to get loss by name"""
    loss_map = {
        'mse': MSELoss,
        'l2': MSELoss,
        'l1': L1Loss,
        'mae': L1Loss,
        'cross_entropy': CrossEntropyLoss,
        'ce': CrossEntropyLoss,
        'bce': BCELoss,
        'binary_cross_entropy': BCELoss,
        'huber': HuberLoss,
        'smooth_l1': SmoothL1Loss,
        'kl_div': KLDivLoss,
        'kldiv': KLDivLoss
    }
    
    if name.lower() not in loss_map:
        raise ValueError(f"Unknown loss function: {name}")
        
    return loss_map[name.lower()](**kwargs)

__all__ = [
    # Classes
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    'HuberLoss',
    'SmoothL1Loss',
    'KLDivLoss',
    
    # Functions
    'binary_cross_entropy',
    'mse_loss',
    'cross_entropy',
    'get_loss_function'
]
