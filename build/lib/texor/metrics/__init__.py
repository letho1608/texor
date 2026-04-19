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


class RMSE(Metric):
    """Root Mean Squared Error"""
    def __init__(self):
        super().__init__()
        self.mse = MSE()

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        self.mse.update(preds, targets)

    def compute(self) -> float:
        return np.sqrt(self.mse.compute())

    def reset(self) -> None:
        self.mse.reset()


class R2Score(Metric):
    """R-squared (coefficient of determination)"""
    def __init__(self):
        super().__init__()
        self.sum_squared_error = 0.0
        self.total = 0
        self.sum_targets = 0.0
        self.sum_squared_targets = 0.0

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        self.sum_squared_error += np.sum((preds - targets) ** 2)
        self.total += preds.size
        self.sum_targets += np.sum(targets)
        self.sum_squared_targets += np.sum(targets ** 2)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        
        mean_targets = self.sum_targets / self.total
        ss_tot = self.sum_squared_targets - self.total * mean_targets ** 2
        ss_res = self.sum_squared_error
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - ss_res / ss_tot

    def reset(self) -> None:
        self.sum_squared_error = 0.0
        self.total = 0
        self.sum_targets = 0.0
        self.sum_squared_targets = 0.0


class ConfusionMatrix(Metric):
    """Confusion Matrix metric"""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = None

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

        if self.matrix is None:
            self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for t, p in zip(targets.flatten(), preds.flatten()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t, p] += 1

    def compute(self) -> np.ndarray:
        if self.matrix is None:
            return np.zeros((self.num_classes, self.num_classes))
        return self.matrix

    def reset(self) -> None:
        self.matrix = None


class AUC(Metric):
    """Area Under the Curve (AUC-ROC)"""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.targets = []
        self.preds = []

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        self.targets.append(targets.flatten())
        self.preds.append(preds.flatten() if preds.ndim == 1 else preds[:, 1].flatten())

    def compute(self) -> float:
        from sklearn.metrics import roc_auc_score
        
        all_targets = np.concatenate(self.targets)
        all_preds = np.concatenate(self.preds)
        
        try:
            return roc_auc_score(all_targets, all_preds)
        except ValueError:
            return 0.0

    def reset(self) -> None:
        self.targets = []
        self.preds = []


class IoU(Metric):
    """Intersection over Union (IoU) for object detection/segmentation"""
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.intersections = 0.0
        self.unions = 0.0

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        # Binary IoU
        preds_binary = (preds > self.threshold).astype(np.float32)
        targets_binary = (targets > self.threshold).astype(np.float32)

        intersection = np.sum(preds_binary * targets_binary)
        union = np.sum(preds_binary) + np.sum(targets_binary) - intersection

        self.intersections += intersection
        self.unions += union

    def compute(self) -> float:
        if self.unions == 0:
            return 0.0
        return self.intersections / self.unions

    def reset(self) -> None:
        self.intersections = 0.0
        self.unions = 0.0


class DiceCoefficient(Metric):
    """Dice Coefficient (F1 score for segmentation)"""
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.intersections = 0.0
        self.total = 0.0

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        preds_binary = (preds > self.threshold).astype(np.float32)
        targets_binary = (targets > self.threshold).astype(np.float32)

        intersection = np.sum(preds_binary * targets_binary)
        self.intersections += intersection
        self.total += np.sum(preds_binary) + np.sum(targets_binary)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return 2 * self.intersections / self.total

    def reset(self) -> None:
        self.intersections = 0.0
        self.total = 0.0


class BLEU(Metric):
    """BLEU score for text generation"""
    def __init__(self, n_gram: int = 4):
        super().__init__()
        self.n_gram = n_gram
        self.references = []
        self.hypotheses = []

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        self.references.append(targets)
        self.hypotheses.append(preds)

    def compute(self) -> float:
        from sklearn.metrics import bleu_score
        
        try:
            return bleu_score(self.references, self.hypotheses)
        except ValueError:
            return 0.0

    def reset(self) -> None:
        self.references = []
        self.hypotheses = []


class Perplexity(Metric):
    """Perplexity for language models"""
    def __init__(self):
        super().__init__()
        self.loss_sum = 0.0
        self.num_tokens = 0

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        # Cross-entropy loss
        ce = -targets * np.log(preds + 1e-10) - (1 - targets) * np.log(1 - preds + 1e-10)
        self.loss_sum += np.sum(ce)
        self.num_tokens += preds.size

    def compute(self) -> float:
        if self.num_tokens == 0:
            return 0.0
        return np.exp(self.loss_sum / self.num_tokens)

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.num_tokens = 0


class MeanAveragePrecision(Metric):
    """Mean Average Precision for object detection"""
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.predictions = []
        self.targets = []

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        if isinstance(preds, Tensor):
            preds = preds.numpy()
        if isinstance(targets, Tensor):
            targets = targets.numpy()

        self.predictions.append(preds)
        self.targets.append(targets)

    def compute(self) -> float:
        # Simplified mAP computation
        # In practice, this would use proper COCO evaluation
        return 0.0

    def reset(self) -> None:
        self.predictions = []
        self.targets = []


class MetricCollection:
    """Collection of metrics for convenient evaluation"""
    def __init__(self, metrics: List[Metric]):
        self.metrics = {m.__class__.__name__: m for m in metrics}

    def update(self, preds: Union[Tensor, np.ndarray],
               targets: Union[Tensor, np.ndarray]) -> None:
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]


def get_metric(name: str, **kwargs) -> Metric:
    """Get metric by name"""
    metrics = {
        'accuracy': Accuracy,
        'precision': Precision,
        'recall': Recall,
        'f1': F1Score,
        'mse': MSE,
        'mae': MAE,
        'rmse': RMSE,
        'r2': R2Score,
        'confusion_matrix': ConfusionMatrix,
        'auc': AUC,
        'iou': IoU,
        'dice': DiceCoefficient,
        'perplexity': Perplexity,
    }
    
    name = name.lower()
    if name not in metrics:
        raise ValueError(f"Unknown metric: {name}")
    
    return metrics[name](**kwargs)