from typing import List, Union, Callable, Dict, Any, Optional, Iterator
import numpy as np
from ..core.native_tensor import Tensor
from .layers import Layer

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._training: bool = True
        self._modules: Dict[str, 'Module'] = {}
        self._parameters: Dict[str, Tensor] = {}
        
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs) -> Tensor:
        """Define the forward pass - must be implemented by subclasses"""
        raise NotImplementedError
        
    def train(self, mode: bool = True) -> 'Module':
        """Set training mode"""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
        
    def eval(self) -> 'Module':
        """Set evaluation mode"""
        return self.train(False)
        
    @property  
    def training(self) -> bool:
        """Check if in training mode"""
        return self._training
        
    def parameters(self) -> Iterator[Tensor]:
        """Iterator over module parameters"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
            
    def named_parameters(self) -> Iterator[tuple]:
        """Iterator over module parameters with names"""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                yield f"{module_name}.{param_name}", param
                
    def zero_grad(self) -> None:
        """Zero gradients for all parameters"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill(0)
                
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

class Model(Module):
    """Base class for all neural network models"""
    
    def __init__(self):
        super().__init__()
        self.layers: List[Layer] = []
        self.optimizer: Optional[Any] = None
        self.loss_fn: Optional[Callable] = None
        self.metrics: List[str] = []
        
    def add(self, layer: Layer) -> None:
        """Add a layer to the model"""
        self.layers.append(layer)
        setattr(self, f'layer_{len(self.layers)-1}', layer)
        
    def train(self, mode: bool = True) -> 'Model':
        """Set training mode for model and all layers"""
        super().train(mode)
        for layer in self.layers:
            if hasattr(layer, 'train'):
                if mode:
                    layer.train()
                else:
                    layer.eval()
            else:
                layer.training = mode
        return self
        
    def eval(self) -> 'Model':
        """Set evaluation mode for model and all layers"""
        return self.train(False)
        
    def parameters(self):
        """Get all parameters from the model and its layers"""
        # First get parameters from parent Module
        for param in super().parameters():
            yield param
        # Then get parameters from all layers
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    yield param
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer(x)
        return x

class Sequential(Model):
    """Sequential container of layers"""
    
    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]
        for layer in layers:
            self.add(layer)
            
    def compile(self, 
                optimizer: Union[str, Any] = 'adam',
                loss: Union[str, Callable] = 'mse',
                metrics: Optional[List[str]] = None) -> None:
        """Configure the model for training"""
        from ..optim import SGD, Adam, RMSprop
        from .loss import get_loss_function
        
        # Set up optimizer
        if isinstance(optimizer, str):
            optimizer_map = {
                'sgd': SGD,
                'adam': Adam,
                'rmsprop': RMSprop
            }
            if optimizer.lower() not in optimizer_map:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            self.optimizer = optimizer_map[optimizer.lower()](list(self.parameters()))
        else:
            self.optimizer = optimizer
        
        # Set up loss function
        self.loss_fn = get_loss_function(loss) if isinstance(loss, str) else loss
            
        # Set up metrics
        self.metrics = metrics or []
            
    def fit(self, 
            x: Union[np.ndarray, Tensor, Any] = None,
            y: Union[np.ndarray, Tensor] = None,
            epochs: int = 1,
            batch_size: int = 32,
            validation_split: float = 0.0,
            verbose: bool = True,
            shuffle: bool = True) -> Dict[str, List[float]]:
        """Train the model
        
        Args:
            x: Input data (numpy array, Tensor, or DataLoader)
            y: Target data (numpy array or Tensor). Not needed if x is a DataLoader.
            epochs: Number of training epochs
            batch_size: Batch size (if x is not a DataLoader)
            validation_split: Fraction of data to use for validation
            verbose: Whether to print progress
            shuffle: Whether to shuffle data (if x is not a DataLoader)
        """
        from ..data.dataset import DataLoader, TensorDataset
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Model must be compiled before training. Call model.compile()")
            
        # Handle DataLoader input
        if isinstance(x, DataLoader):
            train_loader = x
            x_val, y_val = None, None # Validation split not supported for DataLoader yet
        else:
            if not isinstance(x, Tensor):
                x = Tensor(x)
            if not isinstance(y, Tensor):
                y = Tensor(y)
                
            n_samples = x.shape[0]
            
            # Split validation data if needed
            x_val, y_val = None, None
            if validation_split > 0:
                val_size = int(n_samples * validation_split)
                indices = np.arange(n_samples)
                if shuffle:
                    np.random.shuffle(indices)
                train_indices = indices[:-val_size]
                val_indices = indices[-val_size:]
                
                x_val = Tensor(x.data[val_indices])
                y_val = Tensor(y.data[val_indices])
                x_train = Tensor(x.data[train_indices])
                y_train = Tensor(y.data[train_indices])
                
                dataset = TensorDataset(x_train, y_train)
            else:
                dataset = TensorDataset(x, y)
                
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
        history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [] if (validation_split > 0 or x_val is not None) else []
        }
        
        try:
            for epoch in range(epochs):
                # Training
                self.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
                    if len(batch) == 2:
                        x_batch, y_batch = batch
                    else:
                        x_batch = batch[0]
                        y_batch = batch[1] # Assumes last element or similar, but standard is (x, y)
                    
                    # Forward pass
                    y_pred = self.forward(x_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += float(loss.data)
                    num_batches += 1
                    
                epoch_loss /= num_batches
                history['loss'].append(epoch_loss)
                
                # Validation
                val_loss = 0.0
                if x_val is not None:
                    self.eval()
                    val_pred = self.forward(x_val)
                    val_loss = float(self.loss_fn(val_pred, y_val).data)
                    history['val_loss'].append(val_loss)
                    self.train()
                        
                if verbose:
                    status = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}"
                    if x_val is not None:
                        status += f" - val_loss: {val_loss:.4f}"
                    print(status)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            raise
            
        return history
        
    def predict(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Generate predictions for input samples"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
            
        was_training = self.training
        self.eval()
        try:
            predictions = self.forward(x)
        finally:
            if was_training:
                self.train()
            
        return predictions
        
    def evaluate(self, x: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]) -> float:
        """Evaluate the model on test data"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
            
        was_training = self.training
        self.eval()
        try:
            predictions = self.forward(x)
            loss = self.loss_fn(predictions, y)
            return float(loss.data)
        finally:
            if was_training:
                self.train()
        
    def save(self, path: str) -> None:
        """Save model weights to file"""
        import pickle
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod  
    def load(cls, path: str, layers: List[Layer]) -> 'Sequential':
        """Load model weights from file"""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        model = cls(*layers)
        model.load_state_dict(state)
        return model
        
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Get model state dictionary"""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Load model state dictionary"""
        param_dict = dict(self.named_parameters())
        for name, data in state_dict.items():
            if name in param_dict:
                param_dict[name].data = data.copy()

__all__ = [
    'Module',
    'Model', 
    'Sequential'
]