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
            
        pred_data = predictions.data
        target_data = targets.data
        
        # Check if targets are class indices or one-hot
        if target_data.ndim == pred_data.ndim - 1:
            # Class indices - convert to one-hot
            num_classes = pred_data.shape[-1]
            target_one_hot = np.zeros_like(pred_data)
            target_indices = target_data.flatten().astype(int)
            batch_indices = np.arange(target_indices.size)
            target_one_hot.reshape(-1, num_classes)[batch_indices, target_indices] = 1
            target_one_hot = target_one_hot.reshape(pred_data.shape)
            target_data = target_one_hot
        
        # Apply softmax for numerical stability
        exp_pred = np.exp(pred_data - np.max(pred_data, axis=-1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
        
        # Compute cross entropy
        losses = -np.sum(target_data * np.log(softmax_pred + 1e-7), axis=-1)
        
        # Apply weight if provided
        if self.weight is not None:
            if target_data.ndim > 1:
                class_indices = np.argmax(target_data, axis=-1)
                weight_mask = self.weight[class_indices]
                losses = losses * weight_mask
            else:
                losses = losses * self.weight
            
        # Handle ignore_index
        if self.ignore_index >= 0:
            if target_data.ndim > 1:
                mask = np.any(target_data == self.ignore_index, axis=-1)
            else:
                mask = target_data == self.ignore_index
            losses = np.where(mask, 0, losses)
        
        # Create tensor and apply reduction
        loss_tensor = Tensor(losses)
        return self._apply_reduction(loss_tensor)

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
            
        pred_data = predictions.data
        target_data = targets.data
        
        # Clip predictions for numerical stability
        pred_data = np.clip(pred_data, 1e-7, 1 - 1e-7)
        losses = -target_data * np.log(pred_data) - (1 - target_data) * np.log(1 - pred_data)
        
        if self.weight is not None:
            losses = losses * self.weight
        
        # Create tensor and apply reduction    
        loss_tensor = Tensor(losses)
        return self._apply_reduction(loss_tensor)

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
        abs_diff = diff.abs() if hasattr(diff, 'abs') else Tensor(np.abs(diff.data))
        
        return self._apply_reduction(abs_diff)

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
        abs_diff = Tensor(np.abs(diff.data))
        
        quadratic = Tensor(np.minimum(abs_diff.data, self.delta))
        linear = abs_diff - quadratic
        losses = quadratic * quadratic * 0.5 + linear * self.delta
        
        return self._apply_reduction(losses)

class SmoothL1Loss(Loss):
    """Smooth L1 Loss (same as Huber with delta=1.0)"""
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.beta = beta
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
            
        diff = predictions - targets
        abs_diff = Tensor(np.abs(diff.data))
        
        losses = Tensor(np.where(abs_diff.data < self.beta,
                         0.5 * diff.data ** 2 / self.beta,
                         abs_diff.data - 0.5 * self.beta))
        
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
            
        pred_data = predictions.data
        target_data = targets.data
        
        if self.log_target:
            # Both inputs are in log space
            losses = np.exp(target_data) * (target_data - pred_data)
        else:
            # Predictions in log space, targets in probability space
            losses = target_data * (np.log(target_data + 1e-7) - pred_data)
            
        losses = np.sum(losses, axis=-1)
        loss_tensor = Tensor(losses)
        return self._apply_reduction(loss_tensor)

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
