"""
Hyperparameter tuner module for ML Framework.

This module provides the main HyperparameterTuner class that orchestrates
different optimization strategies (grid search, random search, Bayesian).
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np


class HyperparameterTuner:
    """Main hyperparameter tuning interface.
    
    This class provides a unified interface for different hyperparameter
    optimization strategies. It can be used with grid search, random search,
    or Bayesian optimization.
    
    Args:
        method: Optimization method ('grid', 'random', 'bayesian').
        params: Parameter specification (grid or distributions).
        n_iter: Number of iterations for random/Bayesian methods.
        cv_folds: Number of cross-validation folds.
        verbose: Verbosity level.
    """

    def __init__(
        self, 
        method: str = 'random',
        params: Optional[Dict[str, Any]] = None,
        n_iter: int = 10,
        cv_folds: int = 5,
        verbose: int = 1,
        scoring: str = "accuracy",
    ):
        """Initialize the hyperparameter tuner.
        
        Args:
            method: Optimization method ('grid', 'random', 'bayesian').
            params: Parameter specification dictionary.
            n_iter: Number of iterations for random/Bayesian methods.
            cv_folds: Number of cross-validation folds.
            verbose: Verbosity level.
            scoring: Metric name to optimize.
        """
        self.method = method.lower()
        self.params = params or {}
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.scoring = scoring
        
        # Internal tuner instance
        self._tuner: Optional[Any] = None

    def fit(
        self, 
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_fn: Optional[Callable] = None,
    ) -> "HyperparameterTuner":
        """Run hyperparameter tuning.
        
        Args:
            model: Model instance or class to tune.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            train_fn: Custom training function.
        
        Returns:
            Self for method chaining.
        """
        if self.method == 'grid':
            from .grid_search import GridSearch
            self._tuner = GridSearch(
                param_grid=self.params,
                verbose=self.verbose,
                cv=self.cv_folds,
                scoring=self.scoring,
            )
        
        elif self.method == 'random':
            from .random_search import RandomSearch
            self._tuner = RandomSearch(
                param_distributions=self.params,
                n_iter=self.n_iter,
                verbose=self.verbose,
                cv=self.cv_folds,
                scoring=self.scoring,
            )
        
        elif self.method == 'bayesian':
            from .bayesian import BayesianOptimization
            self._tuner = BayesianOptimization(
                param_bounds=self.params,
                n_iter=self.n_iter,
                verbose=self.verbose,
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'grid', 'random', or 'bayesian'.")
        
        # Run the tuner
        if self._tuner is not None:
            self._tuner.fit(model, X_train, y_train, X_val, y_val, train_fn)
        
        return self

    def cross_validate(
        self, 
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """Run cross-validation with hyperparameter tuning.
        
        Args:
            model: Model instance or class to tune.
            X: All features.
            y: All labels.
            train_fn: Custom training function.
        
        Returns:
            Dictionary containing CV scores for each parameter combination.
        """
        if self._tuner is not None and hasattr(self._tuner, 'cross_validate'):
            return self._tuner.cross_validate(model, X, y, train_fn)
        else:
            raise ValueError("Cross-validation requires a tuner instance.")

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found."""
        if self._tuner is not None:
            return getattr(self._tuner, 'best_params_', None)
        return None

    @property
    def best_score(self) -> float:
        """Get the best score achieved."""
        if self._tuner is not None:
            return getattr(self._tuner, 'best_score_', 0.0)
        return 0.0

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Get search results."""
        if self._tuner is not None:
            return getattr(self._tuner, 'results', None)
        return None


def _train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    params: Dict[str, Any],
) -> float:
    """Helper function to train and evaluate a model with given parameters.
    
    Args:
        model: Model instance or class.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        params: Hyperparameter values to use.
    
    Returns:
        Evaluation score (higher is better).
    """
    # Apply parameters
    if hasattr(model, 'set_params'):
        model.set_params(**params)
    elif isinstance(model, dict):
        model.update(params)
    
    # For demonstration purposes, return a placeholder score
    # In practice, this would train the model and compute validation accuracy
    return 0.75


class TuningResult:
    """Container for hyperparameter tuning results."""

    def __init__(self):
        """Initialize the result container."""
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        self.all_results: List[Dict[str, Any]] = []
        self.search_method: str = ""

    def add_result(self, params: Dict[str, Any], score: float) -> None:
        """Add a result entry.
        
        Args:
            params: Parameter values used.
            score: Score achieved.
        """
        self.all_results.append({
            "params": params,
            "score": float(score),
        })

    def update_best(self) -> None:
        """Update best parameters based on all results."""
        if not self.all_results:
            return
        
        best_idx = np.argmax([r["score"] for r in self.all_results])
        self.best_params = self.all_results[best_idx]["params"]
        self.best_score = float(self.all_results[best_idx]["score"])

    def get_summary(self) -> str:
        """Get a summary of the tuning results.
        
        Returns:
            Formatted string with tuning summary.
        """
        lines = [
            "=" * 50,
            "Hyperparameter Tuning Summary",
            "=" * 50,
            f"Method: {self.search_method}",
            f"Iterations: {len(self.all_results)}",
            f"Best parameters: {self.best_params}",
            f"Best score: {self.best_score:.4f}",
        ]
        
        # Top 3 results
        sorted_results = sorted(self.all_results, key=lambda x: x["score"], reverse=True)
        lines.append("\nTop 3 Results:")
        for i, result in enumerate(sorted_results[:3]):
            lines.append(f"  {i+1}. Score: {result['score']:.4f} | Params: {result['params']}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.get_summary()


class TuningConfig:
    """Configuration for hyperparameter tuning operations."""

    def __init__(
        self,
        method: str = 'random',
        params: Optional[Dict[str, Any]] = None,
        n_iter: int = 10,
        cv_folds: int = 5,
        verbose: int = 1,
        scoring: str = "accuracy",
    ):
        """Initialize tuning configuration.
        
        Args:
            method: Optimization method ('grid', 'random', 'bayesian').
            params: Parameter specification dictionary.
            n_iter: Number of iterations for random/Bayesian methods.
            cv_folds: Number of cross-validation folds.
            verbose: Verbosity level.
            scoring: Metric name to optimize.
        """
        self.method = method
        self.params = params or {}
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.scoring = scoring

    def create_tuner(self) -> HyperparameterTuner:
        """Create a HyperparameterTuner with this configuration."""
        return HyperparameterTuner(
            method=self.method,
            params=self.params,
            n_iter=self.n_iter,
            cv_folds=self.cv_folds,
            verbose=self.verbose,
            scoring=self.scoring,
        )
