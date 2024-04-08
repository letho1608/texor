from typing import List, Tuple, Optional
import numpy as np
from ..core.native_tensor import Tensor

# Enhanced optimizer implementations with adaptive learning rates
# Improved convergence properties and numerical stability
# Added support for various optimization strategies

class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr
        
    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        for param in self.params:
            if param.grad is not None:
                param.grad.data.fill(0)
                
    def step(self) -> None:
        """Update parameters using gradients"""
        raise NotImplementedError
        
    def state_dict(self) -> dict:
        """Returns the state of the optimizer"""
        return {'lr': self.lr}
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the optimizer state"""
        self.lr = state_dict['lr']

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01, 
                 momentum: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p.data) for p in params]
        
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            if self.momentum != 0:
                velocity = self.velocities[i]
                velocity = self.momentum * velocity + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * velocity
                else:
                    grad = velocity
                    
                self.velocities[i] = velocity
                
            param.data = param.data - self.lr * grad
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'velocities': [v.copy() for v in self.velocities]
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.momentum = state_dict['momentum']
        self.weight_decay = state_dict['weight_decay']
        self.nesterov = state_dict['nesterov']
        self.velocities = [v.copy() for v in state_dict['velocities']]

class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0,
                 amsgrad: bool = False):
        super().__init__(params, lr)
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("Invalid beta parameters")
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment
        self.v_max = [np.zeros_like(p.data) for p in params] if amsgrad else None
        
    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            if self.amsgrad:
                # Maintain the maximum of all 2nd moment running avg. till now
                self.v_max[i] = np.maximum(self.v_max[i], v_hat)
                denom = np.sqrt(self.v_max[i]) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps
                
            param.data = param.data - self.lr * m_hat / denom
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v],
            'v_max': [v.copy() for v in self.v_max] if self.amsgrad else None
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]
        if self.amsgrad:
            self.v_max = [v.copy() for v in state_dict['v_max']]

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0,
                 centered: bool = False):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        self.square_avg = [np.zeros_like(p.data) for p in params]
        self.momentum_buffer = [np.zeros_like(p.data) for p in params] if momentum != 0 else None
        self.grad_avg = [np.zeros_like(p.data) for p in params] if centered else None
        
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            # Update running average of squared gradients
            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * (grad ** 2)
            
            if self.centered:
                # Update running average of gradients
                self.grad_avg[i] = self.alpha * self.grad_avg[i] + (1 - self.alpha) * grad
                avg = self.square_avg[i] - self.grad_avg[i] ** 2
            else:
                avg = self.square_avg[i]
                
            if self.momentum > 0:
                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + grad / (np.sqrt(avg) + self.eps)
                param.data = param.data - self.lr * self.momentum_buffer[i]
            else:
                param.data = param.data - self.lr * grad / (np.sqrt(avg) + self.eps)
                
    def state_dict(self) -> dict:
        state = {
            **super().state_dict(),
            'alpha': self.alpha,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'centered': self.centered,
            'square_avg': [s.copy() for s in self.square_avg]
        }
        if self.momentum > 0:
            state['momentum_buffer'] = [m.copy() for m in self.momentum_buffer]
        if self.centered:
            state['grad_avg'] = [g.copy() for g in self.grad_avg]
        return state
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.alpha = state_dict['alpha']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.momentum = state_dict['momentum']
        self.centered = state_dict['centered']
        self.square_avg = [s.copy() for s in state_dict['square_avg']]
        if self.momentum > 0:
            self.momentum_buffer = [m.copy() for m in state_dict['momentum_buffer']]
        if self.centered:
            self.grad_avg = [g.copy() for g in state_dict['grad_avg']]

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01,
                 amsgrad: bool = False):
        super().__init__(params, lr)
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("Invalid beta parameters")
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment
        self.v_max = [np.zeros_like(p.data) for p in params] if amsgrad else None
        
    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data.copy()
            
            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            if self.amsgrad:
                # Maintain the maximum of all 2nd moment running avg. till now
                self.v_max[i] = np.maximum(self.v_max[i], v_hat)
                denom = np.sqrt(self.v_max[i]) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps
                
            # Apply weight decay (decoupled)
            param.data = param.data * (1 - self.lr * self.weight_decay)
            
            # Apply gradient update
            param.data = param.data - self.lr * m_hat / denom
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v],
            'v_max': [v.copy() for v in self.v_max] if self.amsgrad else None
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]
        if self.amsgrad:
            self.v_max = [v.copy() for v in state_dict['v_max']]
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.numpy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.numpy()
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * np.square(grad)
            
            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i], v_hat)
                v_hat = self.v_max[i]
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v],
            'v_max': [v.copy() for v in self.v_max] if self.amsgrad else None
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]
        if self.amsgrad:
            self.v_max = [v.copy() for v in state_dict['v_max']]

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0,
                 centered: bool = False):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        self.square_avg = [np.zeros_like(p.numpy()) for p in params]
        self.momentum_buffer = [np.zeros_like(p.numpy()) for p in params] if momentum > 0 else None
        self.grad_avg = [np.zeros_like(p.numpy()) for p in params] if centered else None
        
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.numpy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.numpy()
            
            self.square_avg[i] = self.alpha * self.square_avg[i] + \
                                (1 - self.alpha) * np.square(grad)
            
            if self.centered:
                self.grad_avg[i] = self.alpha * self.grad_avg[i] + \
                                  (1 - self.alpha) * grad
                avg = np.sqrt(self.square_avg[i] - np.square(self.grad_avg[i]) + self.eps)
            else:
                avg = np.sqrt(self.square_avg[i] + self.eps)
            
            if self.momentum > 0:
                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + \
                                        grad / avg
                grad = self.momentum_buffer[i]
            else:
                grad = grad / avg
            
            param.data = param.data - self.lr * grad

class Adagrad(Optimizer):
    """Adagrad optimizer"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 lr_decay: float = 0, weight_decay: float = 0,
                 eps: float = 1e-10):
        super().__init__(params, lr)
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        self.initial_lr = lr
        
        self.step_count = 0
        self.state = [np.zeros_like(p.numpy()) for p in params]
        
    def step(self) -> None:
        self.step_count += 1
        
        # Update learning rate with decay
        if self.lr_decay != 0:
            self.lr = self.initial_lr / (1 + self.lr_decay * self.step_count)
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.numpy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.numpy()
            
            # Update accumulated squared gradients
            self.state[i] = self.state[i] + np.square(grad)
            
            # Update parameters
            param.data = param.data - self.lr * grad / (np.sqrt(self.state[i]) + self.eps)
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'eps': self.eps,
            'initial_lr': self.initial_lr,
            'step_count': self.step_count,
            'state': [s.copy() for s in self.state]
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.lr_decay = state_dict['lr_decay']
        self.weight_decay = state_dict['weight_decay']
        self.eps = state_dict['eps']
        self.initial_lr = state_dict['initial_lr']
        self.step_count = state_dict['step_count']
        self.state = [s.copy() for s in state_dict['state']]

class Adadelta(Optimizer):
    """Adadelta optimizer
    
    It adapts learning rates based on a moving window of gradient updates, instead of
    accumulating all past gradients. This way, Adadelta continues learning even after
    many updates.
    """
    
    def __init__(self, params: List[Tensor], rho: float = 0.9,
                 eps: float = 1e-6, weight_decay: float = 0):
        super().__init__(params, lr=1.0)  # Learning rate is not used in Adadelta
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize accumulated squared gradients and updates
        self.square_avg = [np.zeros_like(p.numpy()) for p in params]
        self.acc_delta = [np.zeros_like(p.numpy()) for p in params]
        
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.numpy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.numpy()
            
            # Update accumulated squared gradients
            self.square_avg[i] = self.rho * self.square_avg[i] + \
                                (1 - self.rho) * np.square(grad)
            
            # Compute update
            std = np.sqrt(self.acc_delta[i] + self.eps)
            delta = np.sqrt(self.square_avg[i] + self.eps)
            update = grad * std / delta
            
            # Update accumulated squared updates
            self.acc_delta[i] = self.rho * self.acc_delta[i] + \
                               (1 - self.rho) * np.square(update)
            
            # Apply update
            param.data = param.data - update
            
    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'rho': self.rho,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'square_avg': [s.copy() for s in self.square_avg],
            'acc_delta': [d.copy() for d in self.acc_delta]
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.rho = state_dict['rho']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.square_avg = [s.copy() for s in state_dict['square_avg']]
        self.acc_delta = [d.copy() for d in state_dict['acc_delta']]

def get_optimizer(name: str) -> type:
    """Get optimizer class by name"""
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
        
    return optimizers[name]