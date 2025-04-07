from typing import List, Union, Callable, Dict, Any, Optional
import numpy as np
from ..core import Tensor
from ..core.backend import backend
from .layers import Layer

class Model:
    """Base class for all neural network models"""
    
    def __init__(self):
        self.layers: List[Layer] = []
        self.training: bool = True
        self.optimizer: Optional[Any] = None
        self.loss_fn: Optional[Callable] = None
        self.metrics: List[str] = []
        
    def add(self, layer: Layer) -> None:
        """Add a layer to the model"""
        self.layers.append(layer)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer(x)
        return x
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
        
    def train(self) -> None:
        """Set model to training mode"""
        self.training = True
        for layer in self.layers:
            layer.train()
            
    def eval(self) -> None:
        """Set model to evaluation mode"""
        self.training = False
        for layer in self.layers:
            layer.eval()
            
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                params.append(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                params.append(layer.bias)
        return params

class Sequential(Model):
    """Sequential container of layers"""
    
    def __init__(self, layers: Optional[List[Layer]] = None):
        super().__init__()
        if layers is not None:
            self.layers = layers
            
    def compile(self, 
                optimizer: Union[str, Any] = 'adam',
                loss: Union[str, Callable] = 'mse',
                metrics: Optional[List[str]] = None) -> None:
        """Configure the model for training"""
        from ..optim import get_optimizer
        from .loss import get_loss_function
        
        # Set up optimizer
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)(self.parameters())
        else:
            self.optimizer = optimizer
        
        # Set up loss function
        self.loss_fn = get_loss_function(loss) if isinstance(loss, str) else loss
            
        # Set up metrics
        self.metrics = metrics or []
            
    def fit(self, 
            x: Union[np.ndarray, Tensor],
            y: Union[np.ndarray, Tensor],
            epochs: int = 1,
            batch_size: int = 32,
            validation_split: float = 0.0,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
            
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Model must be compiled before training. Call model.compile()")
            
        n_samples = len(x)
        indices = np.arange(n_samples)
        
        # Split validation data if needed
        if validation_split > 0:
            val_size = int(n_samples * validation_split)
            train_indices = indices[:-val_size]
            val_indices = indices[-val_size:]
            x_val = x[val_indices]
            y_val = y[val_indices]
            x = x[train_indices]
            y = y[train_indices]
            n_samples = len(x)
            
        history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [] if validation_split > 0 else None
        }
        
        try:
            for epoch in range(epochs):
                # Training
                self.train()
                epoch_loss = 0.0
                np.random.shuffle(indices)
                
                for start_idx in range(0, n_samples, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    x_batch = x[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Forward pass
                    y_pred = self.forward(x_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += float(loss.numpy())
                    
                epoch_loss /= (n_samples // batch_size)
                history['loss'].append(epoch_loss)
                
                # Validation
                if validation_split > 0:
                    self.eval()
                    val_pred = self.forward(x_val)
                    val_loss = float(self.loss_fn(val_pred, y_val).numpy())
                    history['val_loss'].append(val_loss)
                    self.train()
                        
                if verbose:
                    status = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}"
                    if validation_split > 0:
                        status += f" - val_loss: {val_loss:.4f}"
                    print(status)
                    
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            raise
            
        return history
        
    def predict(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Generate predictions for input samples"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        self.eval()
        try:
            predictions = self.forward(x)
        finally:
            self.train()
            
        return predictions
        
    def save(self, path: str) -> None:
        """Save model weights to file"""
        import pickle
        weights = [(name, param.numpy()) for name, param in self.state_dict().items()]
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
            
    @classmethod
    def load(cls, path: str) -> 'Sequential':
        """Load model weights from file"""
        import pickle
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            
        model = cls()
        state_dict = {name: Tensor(array) for name, array in weights}
        model.load_state_dict(state_dict)
        return model
        
    def state_dict(self) -> Dict[str, Tensor]:
        """Get model state dictionary"""
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'state_dict'):
                for name, param in layer.state_dict().items():
                    state[f'layer{i}.{name}'] = param
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load model state dictionary"""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'load_state_dict'):
                prefix = f'layer{i}.'
                layer_state = {
                    name[len(prefix):]: param 
                    for name, param in state_dict.items()
                    if name.startswith(prefix)
                }
                layer.load_state_dict(layer_state)