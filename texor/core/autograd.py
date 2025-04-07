"""Autograd functionality for automatic differentiation"""
from typing import Optional, List, Set
import numpy as np
from typing import TYPE_CHECKING, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor

class Context:
    """Base class for all autograd contexts"""
    def __init__(self, *inputs: 'Tensor'):
        self.inputs = inputs
        self.needs_input_grad = [x.requires_grad for x in inputs]
        
    def backward(self, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Compute gradients with respect to inputs"""
        raise NotImplementedError

class AddBackward(Context):
    """Context for addition backward pass"""
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for addition
        
        If Y = A + B, then:
        dL/dA = dL/dY
        dL/dB = dL/dY
        """
        return grad_output if self.needs_input_grad[0] else None, \
               grad_output if self.needs_input_grad[1] else None

class MatMulBackward(Context):
    """Context for matrix multiplication backward pass"""
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for matrix multiplication
        
        If Y = A @ B, then:
        dL/dA = dL/dY @ B.T
        dL/dB = A.T @ dL/dY
        """
        a, b = self.inputs
        grad_a = np.matmul(grad_output, b.numpy().T) if self.needs_input_grad[0] else None
        grad_b = np.matmul(a.numpy().T, grad_output) if self.needs_input_grad[1] else None
        return grad_a, grad_b

def backward(tensor: 'Tensor',
            gradient: Optional['Tensor'] = None,
            retain_graph: bool = False) -> None:
    """
    Compute gradients of tensor with respect to all tensors that require gradients
    
    Args:
        tensor: Tensor to compute gradients from
        gradient: External gradient to backpropagate, must have same shape as tensor
        retain_graph: Whether to keep computation graph for multiple backward passes
    """
    # Import Tensor here to avoid circular import
    from .tensor import Tensor
    
    # If no external gradient provided, use ones
    if gradient is None:
        gradient = Tensor(1.0) if tensor.shape == () else \
                  Tensor(np.ones_like(tensor.numpy()))
    
    # Initialize gradient buffer
    tensor._grad = gradient
    
    # Build list of nodes in topologically sorted order
    topo_sorted = _build_topo(tensor)
    
    # Backpropagate gradients
    print("\nStarting backpropagation")
    for idx, node in enumerate(topo_sorted):
        print(f"\nProcessing node {idx}:")
        print(f"Node: {node}")
        print(f"Context: {node._ctx.__class__.__name__ if node._ctx else None}")
        
        grad = node._grad
        print(f"Current grad: {grad.numpy() if grad is not None else None}")
        
        # Skip if no gradient
        if grad is None:
            print("Skipping - no gradient")
            continue
            
        # Compute gradients with respect to inputs
        if node._ctx is not None:
            print("Computing backward pass")
            grads = node._ctx.backward(grad.numpy())
            if not isinstance(grads, tuple):
                grads = (grads,)
                
            for input_node, g in zip(node._ctx.inputs, grads):
                if input_node.requires_grad:
                    if input_node._grad is None:
                        input_node._grad = Tensor(g, requires_grad=False)
                    else:
                        # Ensure grad is not tracking gradients
                        input_node._grad = Tensor(
                            input_node._grad.numpy() + g,
                            requires_grad=False
                        )
                        
        # Only clear contexts, keep gradients
        if not retain_graph:
            node._ctx = None

def _build_topo(tensor: 'Tensor') -> List['Tensor']:
    """Build list of all tensors in computation graph in topologically sorted order.
    
    This gives us the order to process gradients, from outputs to inputs."""
    topo: List['Tensor'] = []
    visited: Set['Tensor'] = set()
    
    def build(t: 'Tensor') -> None:
        if t not in visited:
            visited.add(t)
            if t._ctx is not None:
                # Visit input nodes first (deeper in the graph)
                for input_node in reversed(t._ctx.inputs):
                    if input_node not in visited:
                        build(input_node)
            # Add current node after its inputs
            topo.append(t)
            
    build(tensor)
    # Reverse to get correct order (from outputs to inputs)
    topo = list(reversed(topo))
    print("\nTopologically sorted nodes:")
    for idx, t in enumerate(topo):
        print(f"{idx}: {t}, ctx: {t._ctx.__class__.__name__ if t._ctx else None}")
    return topo