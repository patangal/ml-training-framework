"""
Evaluation metrics for ML Framework.

This module provides various evaluation metrics including:
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Regression metrics: MSE, MAE, R²
- Advanced metrics: Matthews Correlation Coefficient, AUC
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np


class Metric(ABC):
    """Abstract base class for evaluation metrics.
    
    All metric classes should inherit from this base class and implement
    the compute method.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the metric.
        
        Args:
            name: Name of the metric (defaults to class name).
        """
        self.name = name or self.__class__.__name__
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        pass

    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None:
        """Update the metric with new predictions and targets.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """Compute the final metric value.
        
        Returns:
            The computed metric value.
        """
        pass

    def __call__(self, predictions: Any, targets: Any) -> float:
        """Update and compute the metric in one call.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
        
        Returns:
            The computed metric value.
        """
        self.update(predictions, targets)
        return self.compute()


class Accuracy(Metric):
    """Classification accuracy metric.
    
    Computes the ratio of correctly predicted observations to total observations.
    Suitable for both binary and multi-class classification.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.correct = 0
        self.total = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update accuracy with new predictions and targets.
        
        Args:
            predictions: Model predictions (class labels).
            targets: Ground truth labels.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        self.correct += np.sum(preds == targs)
        self.total += len(targs)

    def compute(self) -> float:
        """Compute accuracy.
        
        Returns:
            Accuracy value between 0 and 1.
        """
        if self.total == 0:
            return 0.0
        return float(self.correct / self.total)


class Precision(Metric):
    """Precision metric for classification.
    
    Computes the ratio of true positive predictions to all positive predictions.
    Higher precision indicates fewer false positives.
    
    Args:
        average: Type of averaging ('binary', 'macro', 'micro', 'weighted').
        pos_label: Positive class label (for binary classification).
    """

    def __init__(self, name: Optional[str] = None, average: str = "binary", pos_label: int = 1):
        super().__init__(name)
        self.average = average
        self.pos_label = pos_label
        
        # For per-class precision tracking
        self.tp = 0
        self.fp = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.tp = 0
        self.fp = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update precision with new predictions and targets.
        
        Args:
            predictions: Model predictions (class labels).
            targets: Ground truth labels.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        # Binary classification
        if self.average == "binary":
            true_positives = np.sum((preds == self.pos_label) & (targs == self.pos_label))
            false_positives = np.sum((preds == self.pos_label) & (targs != self.pos_label))
            
            self.tp += true_positives
            self.fp += false_positives
        
        # Multi-class: accumulate per-class statistics
        else:
            classes = np.unique(np.concatenate([preds, targs]))
            for cls in classes:
                tp = np.sum((preds == cls) & (targs == cls))
                fp = np.sum((preds == cls) & (targs != cls))
                
                if hasattr(self, f'tp_{cls}'):
                    setattr(self, f'tp_{cls}', getattr(self, f'tp_{cls}') + tp)
                    setattr(self, f'fp_{cls}', getattr(self, f'fp_{cls}') + fp)
                else:
                    setattr(self, f'tp_{cls}', tp)
                    setattr(self, f'fp_{cls}', fp)

    def compute(self) -> float:
        """Compute precision.
        
        Returns:
            Precision value (0 to 1).
        """
        if self.average == "binary":
            if self.tp + self.fp == 0:
                return 0.0
            return float(self.tp / (self.tp + self.fp))
        
        # Macro or micro averaging for multi-class
        precisions = []
        
        if hasattr(self, 'tp_1'):  # Multi-class case
            classes = [attr.split('_')[1] for attr in dir(self) if attr.startswith('tp_')]
            for cls in classes:
                tp = getattr(self, f'tp_{cls}')
                fp = getattr(self, f'fp_{cls}')
                
                if tp + fp > 0:
                    precisions.append(tp / (tp + fp))
        
        if not precisions:
            return 0.0
        
        if self.average == "macro":
            return float(np.mean(precisions))
        elif self.average == "micro":
            total_tp = sum(getattr(self, f'tp_{c}') for c in classes)
            total_fp = sum(getattr(self, f'fp_{c}') for c in classes)
            if total_tp + total_fp == 0:
                return 0.0
            return float(total_tp / (total_tp + total_fp))
        elif self.average == "weighted":
            # Weighted by support (number of true instances)
            weights = []
            for cls in classes:
                support = np.sum(np.array(self.targets) == int(cls)) if hasattr(self, 'targets') else 100
                weights.append(support)
            
            precisions_arr = np.array(precisions)
            return float(np.average(precisions_arr, weights=weights))


class Recall(Metric):
    """Recall (Sensitivity) metric for classification.
    
    Computes the ratio of true positive predictions to all actual positives.
    Higher recall indicates fewer false negatives.
    
    Args:
        average: Type of averaging ('binary', 'macro', 'micro', 'weighted').
        pos_label: Positive class label (for binary classification).
    """

    def __init__(self, name: Optional[str] = None, average: str = "binary", pos_label: int = 1):
        super().__init__(name)
        self.average = average
        self.pos_label = pos_label
        
        self.tp = 0
        self.fn = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.tp = 0
        self.fn = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update recall with new predictions and targets.
        
        Args:
            predictions: Model predictions (class labels).
            targets: Ground truth labels.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        # Binary classification
        if self.average == "binary":
            true_positives = np.sum((preds == self.pos_label) & (targs == self.pos_label))
            false_negatives = np.sum((preds != self.pos_label) & (targs == self.pos_label))
            
            self.tp += true_positives
            self.fn += false_negatives
        
        # Multi-class
        else:
            classes = np.unique(np.concatenate([preds, targs]))
            for cls in classes:
                tp = np.sum((preds == cls) & (targs == cls))
                fn = np.sum((preds != cls) & (targs == cls))
                
                if hasattr(self, f'tp_{cls}'):
                    setattr(self, f'tp_{cls}', getattr(self, f'tp_{cls}') + tp)
                    setattr(self, f'fn_{cls}', getattr(self, f'fn_{cls}') + fn)
                else:
                    setattr(self, f'tp_{cls}', tp)
                    setattr(self, f'fn_{cls}', fn)

    def compute(self) -> float:
        """Compute recall.
        
        Returns:
            Recall value (0 to 1).
        """
        if self.average == "binary":
            if self.tp + self.fn == 0:
                return 0.0
            return float(self.tp / (self.tp + self.fn))
        
        # Multi-class averaging
        recalls = []
        
        if hasattr(self, 'tp_1'):
            classes = [attr.split('_')[1] for attr in dir(self) if attr.startswith('tp_')]
            for cls in classes:
                tp = getattr(self, f'tp_{cls}')
                fn = getattr(self, f'fn_{cls}')
                
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
        
        if not recalls:
            return 0.0
        
        if self.average == "macro":
            return float(np.mean(recalls))
        elif self.average == "micro":
            total_tp = sum(getattr(self, f'tp_{c}') for c in classes)
            total_fn = sum(getattr(self, f'fn_{c}') for c in classes)
            if total_tp + total_fn == 0:
                return 0.0
            return float(total_tp / (total_tp + total_fn))
        elif self.average == "weighted":
            weights = []
            for cls in classes:
                support = np.sum(np.array(self.targets) == int(cls)) if hasattr(self, 'targets') else 100
                weights.append(support)
            
            recalls_arr = np.array(recalls)
            return float(np.average(recalls_arr, weights=weights))


class F1Score(Metric):
    """F1-Score metric for classification.
    
    The harmonic mean of precision and recall. Provides a balance between
    both metrics, especially useful when classes are imbalanced.
    
    Args:
        average: Type of averaging ('binary', 'macro', 'micro', 'weighted').
        pos_label: Positive class label (for binary classification).
    """

    def __init__(self, name: Optional[str] = None, average: str = "binary", pos_label: int = 1):
        super().__init__(name)
        self.average = average
        self.pos_label = pos_label
        
        self.precision_metric = Precision(average=average, pos_label=pos_label)
        self.recall_metric = Recall(average=average, pos_label=pos_label)

    def reset(self) -> None:
        """Reset the metric state."""
        self.precision_metric.reset()
        self.recall_metric.reset()

    def update(self, predictions: Any, targets: Any) -> None:
        """Update F1-score with new predictions and targets.
        
        Args:
            predictions: Model predictions (class labels).
            targets: Ground truth labels.
        """
        self.precision_metric.update(predictions, targets)
        self.recall_metric.update(predictions, targets)

    def compute(self) -> float:
        """Compute F1-score.
        
        Returns:
            F1-Score value (0 to 1).
        """
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        
        if precision + recall == 0:
            return 0.0
        
        return float(2 * (precision * recall) / (precision + recall))


class MeanSquaredError(Metric):
    """Mean Squared Error (MSE) for regression.
    
    Computes the average of squared differences between predictions and targets.
    Sensitive to outliers due to squaring.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_squared_error = 0.0
        self.count = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.sum_squared_error = 0.0
        self.count = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update MSE with new predictions and targets.
        
        Args:
            predictions: Model predictions (continuous values).
            targets: Ground truth values.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        squared_errors = (preds - targs) ** 2
        self.sum_squared_error += np.sum(squared_errors)
        self.count += len(targs)

    def compute(self) -> float:
        """Compute MSE.
        
        Returns:
            Mean Squared Error value.
        """
        if self.count == 0:
            return 0.0
        return float(self.sum_squared_error / self.count)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) for regression.
    
    Computes the average of absolute differences between predictions and targets.
    Less sensitive to outliers than MSE.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.sum_absolute_error = 0.0
        self.count = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.sum_absolute_error = 0.0
        self.count = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update MAE with new predictions and targets.
        
        Args:
            predictions: Model predictions (continuous values).
            targets: Ground truth values.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        absolute_errors = np.abs(preds - targs)
        self.sum_absolute_error += np.sum(absolute_errors)
        self.count += len(targs)

    def compute(self) -> float:
        """Compute MAE.
        
        Returns:
            Mean Absolute Error value.
        """
        if self.count == 0:
            return 0.0
        return float(self.sum_absolute_error / self.count)


class R2Score(Metric):
    """R² (Coefficient of Determination) for regression.
    
    Represents the proportion of variance in the dependent variable that is
    predictable from the independent variables. Range: (-∞, 1].
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.ss_res = 0.0  # Sum of squared residuals
        self.ss_tot = 0.0  # Total sum of squares
        self.count = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.ss_res = 0.0
        self.ss_tot = 0.0
        self.count = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update R² with new predictions and targets.
        
        Args:
            predictions: Model predictions (continuous values).
            targets: Ground truth values.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        self.ss_res += np.sum((targs - preds) ** 2)
        self.ss_tot += np.sum((targs - np.mean(targs)) ** 2)
        self.count += len(targs)

    def compute(self) -> float:
        """Compute R² score.
        
        Returns:
            R² value (can be negative if model performs worse than mean).
        """
        if self.ss_tot == 0:
            return 1.0 if self.ss_res == 0 else 0.0
        
        r2 = 1 - (self.ss_res / self.ss_tot)
        return float(r2)


class MatthewsCorrCoeff(Metric):
    """Matthews Correlation Coefficient (MCC) for binary classification.
    
    A balanced measure that takes into account true and false positives and negatives.
    Suitable for imbalanced datasets. Range: [-1, 1].
    """

    def __init__(self, name: Optional[str] = None, pos_label: int = 1):
        super().__init__(name)
        self.pos_label = pos_label
        
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self) -> None:
        """Reset the metric state."""
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, predictions: Any, targets: Any) -> None:
        """Update MCC with new predictions and targets.
        
        Args:
            predictions: Model predictions (binary labels).
            targets: Ground truth labels.
        """
        if isinstance(predictions, np.ndarray):
            preds = predictions
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, np.ndarray):
            targs = targets
        else:
            targs = np.array(targets)
        
        self.tp += np.sum((preds == self.pos_label) & (targs == self.pos_label))
        self.fp += np.sum((preds == self.pos_label) & (targs != self.pos_label))
        self.tn += np.sum((preds != self.pos_label) & (targs != self.pos_label))
        self.fn += np.sum((preds != self.pos_label) & (targs == self.pos_label))

    def compute(self) -> float:
        """Compute MCC.
        
        Returns:
            Matthews Correlation Coefficient value (-1 to 1).
        """
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * 
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)


class AUC(Metric):
    """Area Under the ROC Curve (AUC) for binary classification.
    
    Measures the ability of a classifier to distinguish between classes.
    Range: [0, 1]. Higher is better.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.predictions = []
        self.targets = []

    def reset(self) -> None:
        """Reset the metric state."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: Any, targets: Any) -> None:
        """Update AUC with new predictions and targets.
        
        Args:
            predictions: Model predictions (probabilities or scores).
            targets: Ground truth labels (binary).
        """
        if isinstance(predictions, np.ndarray):
            self.predictions.extend(predictions.flatten())
        else:
            self.predictions.extend(np.array(predictions).flatten())
        
        if isinstance(targets, np.ndarray):
            self.targets.extend(targets.flatten())
        else:
            self.targets.extend(np.array(targets).flatten())

    def compute(self) -> float:
        """Compute AUC.
        
        Returns:
            Area Under the ROC Curve value (0 to 1).
        """
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return 0.5
        
        from sklearn.metrics import roc_auc_score
        
        try:
            auc = roc_auc_score(self.targets, self.predictions)
            return float(auc)
        except ValueError:
            # Handle edge cases (e.g., all same class)
            return 0.5


class MetricRegistry:
    """Registry for metric classes."""
    
    _metrics: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a metric class.
        
        Args:
            name: The name to register the metric under.
        
        Returns:
            Decorator function.
        """
        def decorator(klass):
            cls._metrics[name] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered metric class by name.
        
        Args:
            name: The name of the metric.
        
        Returns:
            The metric class.
        
        Raises:
            ValueError: If no metric with the given name is registered.
        """
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available metrics: {list(cls._metrics.keys())}")
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics.
        
        Returns:
            List of metric names.
        """
        return list(cls._metrics.keys())


# Register built-in metrics
MetricRegistry.register("accuracy")(Accuracy)
MetricRegistry.register("precision")(Precision)
MetricRegistry.register("recall")(Recall)
MetricRegistry.register("f1")(F1Score)
MetricRegistry.register("mse")(MeanSquaredError)
MetricRegistry.register("mae")(MeanAbsoluteError)
MetricRegistry.register("r2")(R2Score)
MetricRegistry.register("mcc")(MatthewsCorrCoeff)
MetricRegistry.register("auc")(AUC)
