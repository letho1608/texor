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
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta
    }

    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name]


# =============================================================================
# Additional Optimizers
# =============================================================================

class NAdam(Optimizer):
    """Nesterov-accelerated Adaptive Moment Estimation (NAdam) optimizer
    
    Combines Nesterov momentum with Adam's adaptive learning rates.
    """

    def __init__(self, params: List[Tensor], lr: float = 0.002,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment

    def step(self) -> None:
        self.t += 1
        beta1, beta2 = self.betas
        mu_t = beta1 * (1 - 0.5 * 0.96 ** self.t)
        mu_t1 = beta1 * (1 - 0.5 * 0.96 ** (self.t + 1))
        correction = (1 - beta1 ** self.t) * np.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t)
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Compute Nesterov-corrected gradient
            m_hat = (1 - mu_t) * self.m[i] + mu_t1 * grad
            m_hat = m_hat / (1 - beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]


class RAdam(Optimizer):
    """Rectified Adam (RAdam) optimizer
    
    A variant of Adam that rectifies the adaptive learning rate based on
    the variance of the gradients.
    """

    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment

    def step(self) -> None:
        self.t += 1
        beta1, beta2 = self.betas
        
        # Compute the rectification term
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * self.t * (beta2 ** self.t) / (1 - beta2 ** self.t)
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Compute rectified learning rate
            if rho_t > 4:
                r_t = np.sqrt((rho_t - 4) / (rho_inf - 4) * (rho_t - 2) / (rho_inf - 2) * rho_inf / rho_t)
                denom = np.sqrt(v_hat) + self.eps
                param.data = param.data - self.lr * r_t * m_hat / denom
            else:
                denom = np.sqrt(v_hat) + self.eps
                param.data = param.data - self.lr * m_hat / denom

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]


class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)"""

    def __init__(self, params: List[Tensor], lr: float = 0.002,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.u = [np.zeros_like(p.data) for p in params]  # Infinity norm

    def step(self) -> None:
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update the exponentially weighted infinity norm
            self.u[i] = np.maximum(beta2 * self.u[i], np.abs(grad))
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (self.u[i] + self.eps)

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'u': [u.copy() for u in self.u]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.betas = state_dict['betas']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.u = [u.copy() for u in state_dict['u']]


class ASGD(Optimizer):
    """Averaged Stochastic Gradient Descent (ASGD) optimizer"""

    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, weight_decay: float = 0):
        super().__init__(params, lr)
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.t = 0
        self.avg_params = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data.copy()
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Standard SGD update
            param.data = param.data - self.lr * grad
            
            # Update averaged parameters
            if self.t == 1:
                self.avg_params[i] = param.data.copy()
            else:
                self.avg_params[i] = self.alpha * self.avg_params[i] + (1 - self.alpha) * param.data

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'alpha': self.alpha,
            'weight_decay': self.weight_decay,
            't': self.t,
            'avg_params': [a.copy() for a in self.avg_params]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.alpha = state_dict['alpha']
        self.weight_decay = state_dict['weight_decay']
        self.t = state_dict['t']
        self.avg_params = [a.copy() for a in state_dict['avg_params']]


class LBFGS(Optimizer):
    """Limited-memory BFGS (L-BFGS) optimizer
    
    Note: This is a simplified implementation for small datasets.
    """

    def __init__(self, params: List[Tensor], lr: float = 1.0,
                 max_iter: int = 20, max_history: int = 10,
                 line_search: bool = True):
        super().__init__(params, lr)
        self.max_iter = max_iter
        self.max_history = max_history
        self.line_search = line_search
        
        # History of gradient differences and parameter differences
        self.s_history = [[] for _ in params]
        self.y_history = [[] for _ in params]
        self.prev_grads = [np.zeros_like(p.data) for p in params]
        self.prev_params = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data.copy()
            
            # Store initial values
            if len(self.s_history[i]) == 0:
                self.prev_grads[i] = grad.copy()
                self.prev_params[i] = param.data.copy()
                param.data = param.data - self.lr * grad
                continue
            
            # Compute s and y (parameter and gradient differences)
            s = param.data - self.s_history[i][-1] if self.s_history[i] else param.data - self.prev_params[i]
            y = grad - self.prev_grads[i]
            
            # Update history
            if len(self.s_history[i]) >= self.max_history:
                self.s_history[i].pop(0)
                self.y_history[i].pop(0)
            
            self.s_history[i].append(param.data.copy())
            self.y_history[i].append(y.copy())
            
            # Two-loop recursion to compute update direction
            q = grad.copy()
            alphas = []
            
            for s_k, y_k in zip(reversed(self.s_history[i]), reversed(self.y_history[i])):
                rho_k = 1.0 / (np.sum(s_k * y_k) + 1e-10)
                alpha = rho_k * np.sum(s_k * q)
                alphas.append(alpha)
                q = q - alpha * y_k
            
            # Initial Hessian approximation
            s_last = self.s_history[i][-1] if self.s_history[i] else s
            y_last = self.y_history[i][-1] if self.y_history[i] else y
            gamma = np.sum(s_last * y_last) / (np.sum(y_last * y_last) + 1e-10)
            z = gamma * q
            
            for s_k, y_k, alpha in zip(self.s_history[i], self.y_history[i], reversed(alphas)):
                rho_k = 1.0 / (np.sum(s_k * y_k) + 1e-10)
                beta = rho_k * np.sum(y_k * z)
                z = z + s_k * (alpha - beta)
            
            # Update parameters
            param.data = param.data - z
            
            # Store current values
            self.prev_grads[i] = grad.copy()
            self.prev_params[i] = param.data.copy()

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            'max_iter': self.max_iter,
            'max_history': self.max_history,
            'line_search': self.line_search,
            's_history': [s.copy() for s in self.s_history],
            'y_history': [y.copy() for y in self.y_history],
            'prev_grads': [g.copy() for g in self.prev_grads],
            'prev_params': [p.copy() for p in self.prev_params]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.max_iter = state_dict['max_iter']
        self.max_history = state_dict['max_history']
        self.line_search = state_dict['line_search']
        self.s_history = [s.copy() for s in state_dict['s_history']]
        self.y_history = [y.copy() for y in state_dict['y_history']]
        self.prev_grads = [g.copy() for g in state_dict['prev_grads']]
        self.prev_params = [p.copy() for p in state_dict['prev_params']]


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class LRScheduler:
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.lr]

    def step(self) -> None:
        """Update learning rate"""
        raise NotImplementedError

    def get_lr(self) -> List[float]:
        """Get current learning rates"""
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs"""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            for i, param_group in enumerate([self.optimizer.params]):
                param_group.lr = self.base_lrs[i] * (self.gamma ** (self.last_epoch // self.step_size))

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** (self.last_epoch // self.step_size)) for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    """Decays the learning rate by gamma at specified milestones"""

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1,
                 last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for i in range(len(self.optimizer.params)):
                self.optimizer.lr = self.base_lrs[i] * (self.gamma ** self.milestones.index(self.last_epoch))

    def get_lr(self) -> List[float]:
        count = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [base_lr * (self.gamma ** count) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Decays the learning rate by gamma each epoch"""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.95,
                 last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        for i in range(len(self.optimizer.params)):
            self.optimizer.lr = self.base_lrs[i] * (self.gamma ** self.last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule"""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0,
                 last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        for i in range(len(self.optimizer.params)):
            self.optimizer.lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2

    def get_lr(self) -> List[float]:
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving"""

    def __init__(self, optimizer: Optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 0,
                 last_epoch: int = -1):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        super().__init__(optimizer, last_epoch)

    def step(self, metrics: Optional[float] = None) -> None:
        if metrics is None:
            return
            
        self.last_epoch += 1
        
        if self.mode == 'min':
            is_better = metrics < (self.best - self.threshold)
        else:
            is_better = metrics > (self.best + self.threshold)
        
        if is_better:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            for i in range(len(self.optimizer.params)):
                new_lr = max(self.base_lrs[i] * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.base_lrs[i] = new_lr
            self.num_bad_epochs = 0

    def get_lr(self) -> List[float]:
        return [self.optimizer.lr]


class WarmupScheduler(LRScheduler):
    """Learning rate warmup scheduler"""

    def __init__(self, optimizer: Optimizer, warmup_epochs: int,
                 base_scheduler: Optional[LRScheduler] = None,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            for i in range(len(self.optimizer.params)):
                self.optimizer.lr = self.base_lrs[i] * warmup_factor
        elif self.base_scheduler is not None:
            self.base_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            self.base_scheduler.step()
            for i in range(len(self.optimizer.params)):
                self.optimizer.lr = self.base_scheduler.optimizer.lr

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        elif self.base_scheduler is not None:
            return self.base_scheduler.get_lr()
        return self.base_lrs


class CyclicLR(LRScheduler):
    """Cyclical learning rate schedule"""

    def __init__(self, optimizer: Optimizer, base_lr: float, max_lr: float,
                 step_size: int = 2000, mode: str = 'triangular',
                 gamma: float = 1.0, last_epoch: int = -1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        
        cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
        x = np.abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = self.gamma ** self.last_epoch
        else:
            scale_factor = 1.0
        
        delta = self.max_lr - self.base_lr
        self.optimizer.lr = self.base_lr + delta * scale_factor * np.maximum(0, 1 - x)

    def get_lr(self) -> List[float]:
        return [self.optimizer.lr]


class OneCycleLR(LRScheduler):
    """One cycle learning rate schedule
    
    A schedule that starts with a warmup, then decreases the learning rate.
    """

    def __init__(self, optimizer: Optimizer, max_lr: float,
                 total_steps: Optional[int] = None,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 10000.0, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.last_epoch += 1
        
        if self.total_steps is None:
            raise ValueError("total_steps must be specified")
        
        step = self.last_epoch
        pct = step / self.total_steps
        
        if pct < self.pct_start:
            # Warmup phase
            lr_factor = pct / self.pct_start
        else:
            # Decay phase
            lr_factor = 1 - (pct - self.pct_start) / (1 - self.pct_start)
        
        min_lr = self.max_lr / self.div_factor
        final_lr = self.max_lr / self.final_div_factor
        
        self.optimizer.lr = min_lr + (self.max_lr - min_lr) * lr_factor

    def get_lr(self) -> List[float]:
        return [self.optimizer.lr]


def get_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    """Get learning rate scheduler by name"""
    schedulers = {
        'step': StepLR,
        'multistep': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'warmup': WarmupScheduler,
        'cyclic': CyclicLR,
        'onecycle': OneCycleLR
    }

    name = name.lower()
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")

    return schedulers[name](optimizer, **kwargs)