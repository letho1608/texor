"""Utility functions for Texor

This module provides various utility functions for model training,
visualization, checkpointing, and other common tasks.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import os
import json
import pickle
from pathlib import Path

from ..core import Tensor


# =============================================================================
# Checkpointing and Model Saving
# =============================================================================

def save_checkpoint(model: 'Layer', optimizer: 'Optimizer', 
                    epoch: int, loss: float, filepath: str,
                    additional_data: Optional[Dict] = None) -> None:
    """Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        additional_data: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else {},
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else {},
    }
    
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filepath: str, model: 'Layer', 
                    optimizer: Optional['Optimizer'] = None) -> Dict:
    """Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    if hasattr(model, 'load_state_dict') and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and hasattr(optimizer, 'load_state_dict'):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_model(model: 'Layer', filepath: str) -> None:
    """Save only model weights
    
    Args:
        model: Model to save
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else {}
    
    with open(filepath, 'wb') as f:
        pickle.dump(state_dict, f)


def load_model(model: 'Layer', filepath: str) -> None:
    """Load only model weights
    
    Args:
        model: Model to load weights into
        filepath: Path to model file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
    
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(state_dict)


# =============================================================================
# Learning Rate Schedulers (Functional)
# =============================================================================

class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model: 'Layer', decay: float = 0.9999):
        """
        Args:
            model: Model to track
            decay: Decay rate for EMA
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters() if hasattr(model, 'named_parameters') else []:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update EMA parameters"""
        for name, param in self.model.named_parameters() if hasattr(self.model, 'named_parameters') else []:
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters() if hasattr(self.model, 'named_parameters') else []:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """Restore original parameters"""
        for name, param in self.model.named_parameters() if hasattr(self.model, 'named_parameters') else []:
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Gradient Clipping
# =============================================================================

def clip_gradients(model: 'Layer', max_norm: float = 1.0,
                   norm_type: float = 2.0) -> float:
    """Clip gradients by global norm
    
    Args:
        model: Model whose gradients to clip
        max_norm: Maximum norm value
        norm_type: Type of norm (1.0, 2.0, or 'inf')
    
    Returns:
        Total gradient norm before clipping
    """
    total_norm = 0.0
    
    params = list(model.parameters()) if hasattr(model, 'parameters') else []
    
    for p in params:
        if p.grad is not None:
            if norm_type == 'inf':
                param_norm = p.grad.data.abs().max()
            else:
                param_norm = np.linalg.norm(p.grad.data.flatten(), ord=norm_type)
            total_norm += param_norm ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data * clip_coef
    
    return total_norm


# =============================================================================
# Device Management
# =============================================================================

def get_default_device() -> str:
    """Get default device (cuda if available, else cpu)"""
    try:
        import cupy as cp
        return 'cuda'
    except ImportError:
        return 'cpu'


def to_device(data: Union[Tensor, List, Tuple], device: str) -> Union[Tensor, List, Tuple]:
    """Move data to device
    
    Args:
        data: Data to move
        device: Target device
    
    Returns:
        Data on target device
    """
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(d, device) for d in data)
    else:
        return data


class DeviceDataLoader:
    """Data loader that automatically moves data to device"""
    
    def __init__(self, dataloader: 'DataLoader', device: str):
        self.dataloader = dataloader
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch, self.device)
    
    def __len__(self):
        return len(self.dataloader)


# =============================================================================
# Training Helpers
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressBar:
    """Simple progress bar for training"""
    
    def __init__(self, total: int, desc: str = '', length: int = 50):
        self.total = total
        self.desc = desc
        self.length = length
        self.current = 0
    
    def update(self, current: Optional[int] = None, loss: Optional[float] = None) -> None:
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        filled = int(self.length * self.current / self.total)
        bar = '=' * filled + '-' * (self.length - filled)
        
        loss_str = f' loss={loss:.4f}' if loss is not None else ''
        print(f'\r{self.desc} [{bar}] {self.current}/{self.total}{loss_str}', end='')
        
        if self.current >= self.total:
            print()


def count_parameters(model: 'Layer') -> int:
    """Count total number of trainable parameters in model"""
    params = list(model.parameters()) if hasattr(model, 'parameters') else []
    return sum(p.numel() for p in params if p.requires_grad)


def model_summary(model: 'Layer', input_size: Optional[Tuple] = None) -> Dict:
    """Get model summary information
    
    Args:
        model: Model to summarize
        input_size: Input size (optional)
    
    Returns:
        Dictionary with model information
    """
    summary = {
        'total_params': 0,
        'trainable_params': 0,
        'layers': []
    }
    
    layers = list(model.modules()) if hasattr(model, 'modules') else [model]
    
    for layer in layers:
        layer_info = {
            'name': layer.__class__.__name__,
            'params': 0,
            'trainable': False
        }
        
        if hasattr(layer, 'parameters'):
            params = list(layer.parameters())
            layer_info['params'] = sum(p.numel() for p in params)
            layer_info['trainable'] = any(p.requires_grad for p in params)
            summary['trainable_params'] += sum(p.numel() for p in params if p.requires_grad)
        
        summary['total_params'] += layer_info['params']
        
        if layer_info['params'] > 0:
            summary['layers'].append(layer_info)
    
    return summary


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], 
                          metrics: List[str] = ['loss', 'accuracy'],
                          save_path: Optional[str] = None) -> None:
    """Plot training history
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label=metric)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'Training {metric}')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    except ImportError:
        print("matplotlib not available for plotting")


def visualize_predictions(images: Tensor, predictions: Tensor, 
                         targets: Tensor, num_samples: int = 8,
                         class_names: Optional[List[str]] = None) -> None:
    """Visualize model predictions
    
    Args:
        images: Input images
        predictions: Model predictions
        targets: Ground truth labels
        num_samples: Number of samples to visualize
        class_names: List of class names
    """
    try:
        import matplotlib.pyplot as plt
        
        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
        
        for i in range(num_samples):
            # Denormalize image if needed
            img = images[i]
            if img.shape[0] == 3:  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            
            # Clip to valid range
            img = np.clip(img, 0, 1)
            
            # Show image
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            
            # Get predicted and true class
            pred_class = np.argmax(predictions[i])
            true_class = np.argmax(targets[i]) if targets[i].ndim > 0 else int(targets[i])
            
            pred_name = class_names[pred_class] if class_names else str(pred_class)
            true_name = class_names[true_class] if class_names else str(true_class)
            
            color = 'green' if pred_class == true_class else 'red'
            axes[0, i].set_title(f'Pred: {pred_name}', color=color)
            
            # Show ground truth in bottom row
            axes[1, i].text(0.5, 0.5, f'True: {true_name}', 
                           ha='center', va='center', fontsize=12)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.close()
    except ImportError:
        print("matplotlib not available for visualization")


# =============================================================================
# Data Augmentation Utilities
# =============================================================================

def mixup_data(x: Tensor, y: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply mixup data augmentation
    
    Args:
        x: Input tensor
        y: Target tensor
        alpha: Mixup alpha parameter
    
    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x.data + (1 - lam) * x.data[index]
    y_a, y_b = y, y[index]
    
    return Tensor(mixed_x, requires_grad=x.requires_grad), y_a, y_b, lam


def cutmix_data(x: Tensor, y: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply cutmix data augmentation
    
    Args:
        x: Input tensor
        y: Target tensor
        alpha: Cutmix alpha parameter
    
    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    # Get random box
    W = x.shape[-1]
    H = x.shape[-2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    mixed_x = x.data.copy()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x.data[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return Tensor(mixed_x, requires_grad=x.requires_grad), y_a, y_b, lam


def mixup_criterion(criterion: Callable, pred: Tensor, y_a: Tensor, 
                    y_b: Tensor, lam: float) -> Tensor:
    """Compute loss for mixup
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First target
        y_b: Second target
        lam: Mixup lambda
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Memory Management
# =============================================================================

def clear_memory() -> None:
    """Clear GPU memory if using CUDA"""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass
    
    import gc
    gc.collect()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {'cpu': 0}
    
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        stats['gpu_used'] = mempool.used_bytes() / (1024 ** 3)  # GB
        stats['gpu_total'] = mempool.total_bytes() / (1024 ** 3)  # GB
    except ImportError:
        pass
    
    return stats


# =============================================================================
# Logging Utilities
# =============================================================================

class TensorBoardLogger:
    """Simple tensorboard logger"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Try to import tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            self.writer = None
            self.available = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.available:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: Tensor, step: int) -> None:
        if self.available:
            self.writer.add_histogram(tag, values.numpy(), step)
    
    def log_image(self, tag: str, image: Tensor, step: int) -> None:
        if self.available:
            self.writer.add_image(tag, image.numpy(), step)
    
    def close(self) -> None:
        if self.available:
            self.writer.close()


class CSVLogger:
    """CSV logger for training metrics"""
    
    def __init__(self, filepath: str, fields: List[str]):
        self.filepath = filepath
        self.fields = fields
        
        # Create file with header if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(','.join(fields) + '\n')
    
    def log(self, values: Dict[str, float], step: int) -> None:
        row = [str(step)] + [str(values.get(f, '')) for f in self.fields]
        with open(self.filepath, 'a') as f:
            f.write(','.join(row) + '\n')


# =============================================================================
# Misc Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    # Note: For full reproducibility, you would also need to set
    # environment variables for other libraries


def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dest: str) -> None:
    """Download file from URL"""
    import urllib.request
    
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete!")


def calculate_flops(model: 'Layer', input_size: Tuple) -> int:
    """Estimate FLOPs for a model (rough estimate)"""
    # This is a very rough estimate
    total_params = count_parameters(model)
    # Assume each parameter participates in roughly one multiply-add per forward pass
    return total_params * 2  # multiply and add


def freeze(model: 'Layer') -> None:
    """Freeze all parameters in model"""
    for param in model.parameters() if hasattr(model, 'parameters') else []:
        param.requires_grad = False


def unfreeze(model: 'Layer') -> None:
    """Unfreeze all parameters in model"""
    for param in model.parameters() if hasattr(model, 'parameters') else []:
        param.requires_grad = True