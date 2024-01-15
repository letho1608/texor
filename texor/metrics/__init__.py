from typing import List, Optional, Union, Dict, Any, Callable
import numpy as np
from ..core import Tensor

class Metric:
    """Base class for all metrics"""
    def __init__(self):
        self.reset()
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        raise NotImplementedError
        
    def compute(self) -> float:
        raise NotImplementedError
        
    def reset(self) -> None:
        raise NotImplementedError

class Accuracy(Metric):
    """Calculates accuracy for classification tasks"""
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()
            
        if preds.shape != targets.shape:
            preds = np.argmax(preds, axis=-1)
            if len(targets.shape) > 1 and targets.shape[-1] > 1:
                targets = np.argmax(targets, axis=-1)
                
        self.correct += np.sum(preds == targets)
        self.total += targets.size
        
    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
        
    def reset(self) -> None:
        self.correct = 0
        self.total = 0

class Precision(Metric):
    """Calculates precision for binary or multiclass classification"""
    def __init__(self, num_classes: int = 2, average: str = 'macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.true_positives = np.zeros(num_classes)
        self.predicted_positives = np.zeros(num_classes)
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()
            
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            preds = np.argmax(preds, axis=-1)
        if len(targets.shape) > 1 and targets.shape[-1] > 1:
            targets = np.argmax(targets, axis=-1)
            
        for i in range(self.num_classes):
            self.true_positives[i] += np.sum((preds == i) & (targets == i))
            self.predicted_positives[i] += np.sum(preds == i)
            
    def compute(self) -> float:
        precisions = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.predicted_positives[i] > 0:
                precisions[i] = self.true_positives[i] / self.predicted_positives[i]
                
        if self.average == 'macro':
            return np.mean(precisions)
        elif self.average == 'micro':
            return np.sum(self.true_positives) / np.sum(self.predicted_positives)
        else:
            raise ValueError(f"Unsupported average type: {self.average}")
            
    def reset(self) -> None:
        self.true_positives = np.zeros(self.num_classes)
        self.predicted_positives = np.zeros(self.num_classes)

class Recall(Metric):
    """Calculates recall for binary or multiclass classification"""
    def __init__(self, num_classes: int = 2, average: str = 'macro'):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.true_positives = np.zeros(num_classes)
        self.actual_positives = np.zeros(num_classes)
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()
            
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            preds = np.argmax(preds, axis=-1)
        if len(targets.shape) > 1 and targets.shape[-1] > 1:
            targets = np.argmax(targets, axis=-1)
            
        for i in range(self.num_classes):
            self.true_positives[i] += np.sum((preds == i) & (targets == i))
            self.actual_positives[i] += np.sum(targets == i)
            
    def compute(self) -> float:
        recalls = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.actual_positives[i] > 0:
                recalls[i] = self.true_positives[i] / self.actual_positives[i]
                
        if self.average == 'macro':
            return np.mean(recalls)
        elif self.average == 'micro':
            return np.sum(self.true_positives) / np.sum(self.actual_positives)
        else:
            raise ValueError(f"Unsupported average type: {self.average}")
            
    def reset(self) -> None:
        self.true_positives = np.zeros(self.num_classes)
        self.actual_positives = np.zeros(self.num_classes)

class F1Score(Metric):
    """Calculates F1 score"""
    def __init__(self, num_classes: int = 2, average: str = 'macro'):
        super().__init__()
        self.precision = Precision(num_classes, average)
        self.recall = Recall(num_classes, average)
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        
    def compute(self) -> float:
        precision = self.precision.compute()
        recall = self.recall.compute()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
        
    def reset(self) -> None:
        self.precision.reset()
        self.recall.reset()

class MSE(Metric):
    """Mean Squared Error"""
    def __init__(self):
        super().__init__()
        self.sum_squared_error = 0.0
        self.total = 0
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()
            
        self.sum_squared_error += np.sum((preds - targets) ** 2)
        self.total += preds.size
        
    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.sum_squared_error / self.total
        
    def reset(self) -> None:
        self.sum_squared_error = 0.0
        self.total = 0

class MAE(Metric):
    """Mean Absolute Error"""
    def __init__(self):
        super().__init__()
        self.sum_absolute_error = 0.0
        self.total = 0
        
    def update(self, preds: Union[Tensor, np.ndarray], 
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()
            
        self.sum_absolute_error += np.sum(np.abs(preds - targets))
        self.total += preds.size
        
    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.sum_absolute_error / self.total
        
    def reset(self) -> None:
        self.sum_absolute_error = 0.0
        self.total = 0