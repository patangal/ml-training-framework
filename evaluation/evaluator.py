"""
Evaluator class for ML Framework.

This module provides a unified interface for evaluating models using various metrics,
generating reports, and visualizing results.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import json


class Evaluator:
    """Model evaluator that computes and tracks multiple metrics.
    
    This class provides a unified interface for evaluating model performance using
    various metrics. It supports batch-wise computation, epoch-level aggregation,
    and result visualization.
    
    Args:
        metrics: List of metric instances to compute during evaluation.
        verbose: Verbosity level (0 = silent, 1 = display results).
    """

    def __init__(self, metrics: Optional[List[Any]] = None, verbose: int = 1):
        """Initialize the evaluator.
        
        Args:
            metrics: List of metric instances to compute.
            verbose: Verbosity level for output.
        """
        self.metrics = metrics or []
        self.verbose = verbose
        
        # Initialize metric states
        self._metric_states: Dict[str, Any] = {}

    def add_metric(self, metric: Any) -> None:
        """Add a metric to the evaluator.
        
        Args:
            metric: Metric instance to add.
        """
        if metric not in self.metrics:
            self.metrics.append(metric)
    
    def remove_metric(self, name: str) -> bool:
        """Remove a metric by name.
        
        Args:
            name: Name of the metric to remove.
        
        Returns:
            True if metric was removed, False otherwise.
        """
        for i, metric in enumerate(self.metrics):
            if metric.name == name:
                self.metrics.pop(i)
                return True
        return False

    def evaluate(
        self, 
        predictions: Any, 
        targets: Any,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model performance on given data.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels/values.
            batch_size: Batch size for processing (optional).
        
        Returns:
            Dictionary containing metric values.
        """
        results = {}
        
        # Process in batches if specified
        if batch_size is not None and len(predictions) > batch_size:
            num_batches = len(predictions) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                
                pred_batch = predictions[start_idx:end_idx]
                target_batch = targets[start_idx:end_idx]
                
                metric_results = self._compute_metrics(pred_batch, target_batch)
                for key, value in metric_results.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
            
            # Process remaining samples
            if num_batches * batch_size < len(predictions):
                start_idx = num_batches * batch_size
                pred_batch = predictions[start_idx:]
                target_batch = targets[start_idx:]
                
                metric_results = self._compute_metrics(pred_batch, target_batch)
                for key, value in metric_results.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
        else:
            # Process all at once
            results = self._compute_metrics(predictions, targets)
        
        # Average batch-wise results
        for key, values in results.items():
            if isinstance(values, list):
                results[key] = float(np.mean(values))
        
        return results

    def _compute_metrics(
        self, 
        predictions: Any, 
        targets: Any
    ) -> Dict[str, float]:
        """Compute all metrics for given predictions and targets.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels/values.
        
        Returns:
            Dictionary containing metric values.
        """
        results = {}
        
        for metric in self.metrics:
            try:
                # Reset metric before computing on new data
                metric.reset()
                
                # Update and compute
                value = metric(predictions, targets)
                results[metric.name] = float(value)
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error computing {metric.name}: {e}")
                results[metric.name] = float('nan')
        
        return results

    def evaluate_batch(
        self, 
        predictions: Any, 
        targets: Any
    ) -> Dict[str, float]:
        """Evaluate on a single batch of data.
        
        Args:
            predictions: Batch of model predictions.
            targets: Batch of ground truth labels/values.
        
        Returns:
            Dictionary containing metric values for this batch.
        """
        return self._compute_metrics(predictions, targets)

    def evaluate_epoch(
        self, 
        all_predictions: List[Any], 
        all_targets: List[Any]
    ) -> Dict[str, float]:
        """Aggregate metrics over an entire epoch.
        
        Args:
            all_predictions: List of predictions from each batch in the epoch.
            all_targets: List of targets from each batch in the epoch.
        
        Returns:
            Dictionary containing aggregated metric values.
        """
        # Concatenate all batches
        if isinstance(all_predictions[0], np.ndarray):
            combined_preds = np.concatenate(all_predictions)
            combined_targs = np.concatenate(all_targets)
        else:
            combined_preds = np.array(all_predictions).flatten()
            combined_targs = np.array(all_targets).flatten()
        
        return self.evaluate(combined_preds, combined_targs)

    def get_report(self, results: Dict[str, float]) -> str:
        """Generate a formatted evaluation report.
        
        Args:
            results: Dictionary of metric values.
        
        Returns:
            Formatted string containing the evaluation report.
        """
        lines = ["=" * 50, "Evaluation Report", "=" * 50]
        
        for name, value in sorted(results.items()):
            if isinstance(value, float):
                if value == float('nan'):
                    formatted_value = "N/A"
                elif value >= 1:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            lines.append(f"{name:20s}: {formatted_value}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

    def print_report(self, results: Dict[str, float]) -> None:
        """Print the evaluation report.
        
        Args:
            results: Dictionary of metric values.
        """
        if self.verbose > 0:
            print(self.get_report(results))

    def save_results(
        self, 
        results: Dict[str, float], 
        filepath: str,
        append: bool = False
    ) -> None:
        """Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary of metric values.
            filepath: Path to save the results.
            append: Whether to append to existing file.
        """
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        if append and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []
        
        # Add timestamp (optional)
        result_entry = results.copy()
        history.append(result_entry)
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def get_best_metric(
        self, 
        results: Dict[str, float], 
        metric_name: str,
        mode: str = "max"  # or "min"
    ) -> Optional[float]:
        """Get the best value for a specific metric.
        
        Args:
            results: Dictionary of metric values.
            metric_name: Name of the metric to query.
            mode: Whether higher is better ("max") or lower is better ("min").
        
        Returns:
            Best value found, or None if not available.
        """
        if metric_name not in results:
            return None
        
        value = results[metric_name]
        
        if mode == "max":
            # For metrics where higher is better (accuracy, F1, etc.)
            if value > 0 and not np.isnan(value):
                return value
        elif mode == "min":
            # For metrics where lower is better (loss, error)
            if value >= 0 and not np.isnan(value):
                return value
        
        return None


class ClassificationReport:
    """Generate detailed classification reports.
    
    Provides comprehensive analysis including per-class precision, recall,
    F1-score, and support counts.
    """

    def __init__(self, num_classes: int = 10):
        """Initialize the report generator.
        
        Args:
            num_classes: Number of classes in the classification task.
        """
        self.num_classes = num_classes
        
        # Per-class statistics
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, predictions: Any, targets: Any) -> None:
        """Update confusion matrix with new predictions and targets.
        
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
        
        # Update confusion matrix
        for pred, true in zip(preds, targs):
            self.confusion_matrix[int(pred), int(true)] += 1

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute per-class metrics.
        
        Returns:
            Dictionary containing precision, recall, F1-score for each class.
        """
        # Avoid division by zero
        col_sums = self.confusion_matrix.sum(axis=0)
        row_sums = self.confusion_matrix.sum(axis=1)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(self.num_classes):
            # Precision: TP / (TP + FP)
            tp = self.confusion_matrix[i, i]
            fp = col_sums[i] - tp
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            # Recall: TP / (TP + FN)
            fn = row_sums[i] - tp
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            # F1-Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            "per_class_precision": precisions,
            "per_class_recall": recalls,
            "per_class_f1": f1_scores,
            "support": row_sums.tolist(),
        }

    def generate_report(self) -> str:
        """Generate a formatted classification report.
        
        Returns:
            Formatted string containing the classification report.
        """
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 70,
            "Classification Report",
            "=" * 70,
            f"{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}",
            "-" * 70,
        ]
        
        for i in range(self.num_classes):
            lines.append(
                f"{i:<10} {metrics['per_class_precision'][i]:>12.4f} "
                f"{metrics['per_class_recall'][i]:>12.4f} "
                f"{metrics['per_class_f1'][i]:>12.4f} "
                f"{int(metrics['support'][i]):>10}"
            )
        
        # Averages
        avg_precision = np.mean(metrics['per_class_precision'])
        avg_recall = np.mean(metrics['per_class_recall'])
        avg_f1 = np.mean(metrics['per_class_f1'])
        
        lines.extend([
            "-" * 70,
            f"{'Accuracy':<10} {'N/A':>12} {self._compute_accuracy():>12.4f} "
            f"{'N/A':>12} {sum(metrics['support']):>10}",
            f"{'Macro Avg':<10} {avg_precision:>12.4f} {avg_recall:>12.4f} "
            f"{avg_f1:>12.4f} {'N/A':>10}",
            "=" * 70,
        ])
        
        return "\n".join(lines)

    def _compute_accuracy(self) -> float:
        """Compute overall accuracy from confusion matrix."""
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        
        correct = np.trace(self.confusion_matrix)
        return correct / total


class RegressionReport:
    """Generate detailed regression reports.
    
    Provides analysis including MSE, MAE, R² score, and residual statistics.
    """

    def __init__(self):
        """Initialize the regression report generator."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: Any, targets: Any) -> None:
        """Update with new predictions and targets.
        
        Args:
            predictions: Model predictions (continuous values).
            targets: Ground truth values.
        """
        if isinstance(predictions, np.ndarray):
            self.predictions.extend(predictions.flatten())
        else:
            self.predictions.extend(np.array(predictions).flatten())
        
        if isinstance(targets, np.ndarray):
            self.targets.extend(targets.flatten())
        else:
            self.targets.extend(np.array(targets).flatten())

    def compute_metrics(self) -> Dict[str, float]:
        """Compute regression metrics.
        
        Returns:
            Dictionary containing MSE, MAE, R² score.
        """
        preds = np.array(self.predictions)
        targs = np.array(self.targets)
        
        mse = np.mean((preds - targs) ** 2)
        mae = np.mean(np.abs(preds - targs))
        
        ss_res = np.sum((targs - preds) ** 2)
        ss_tot = np.sum((targs - np.mean(targs)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "rmse": float(np.sqrt(mse)),
        }

    def generate_report(self) -> str:
        """Generate a formatted regression report.
        
        Returns:
            Formatted string containing the regression report.
        """
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 50,
            "Regression Report",
            "=" * 50,
            f"{'Metric':<20} {'Value':>15}",
            "-" * 50,
            f"{'MSE':<20} {metrics['mse']:>15.6f}",
            f"{'RMSE':<20} {metrics['rmse']:>15.6f}",
            f"{'MAE':<20} {metrics['mae']:>15.6f}",
            f"{'R² Score':<20} {metrics['r2']:>15.4f}",
            "=" * 50,
        ]
        
        return "\n".join(lines)
