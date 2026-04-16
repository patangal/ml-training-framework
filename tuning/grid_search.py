"""
Grid Search hyperparameter optimization for ML Framework.

This module provides exhaustive grid search over a specified parameter space,
evaluating all possible combinations of hyperparameters.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import itertools


class GridSearch:
    """Exhaustive grid search for hyperparameter optimization.
    
    Evaluates all possible combinations of hyperparameters from the specified grids.
    Suitable when the parameter space is small and exhaustive search is feasible.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values to try.
        n_jobs: Number of parallel jobs (not implemented in basic version).
        verbose: Verbosity level (0 = silent, 1 = progress, 2 = detailed).
        cv: Cross-validation fold count or CV object.
    """

    def __init__(
        self, 
        param_grid: Dict[str, List[Any]],
        n_jobs: int = 1,
        verbose: int = 1,
        cv: int = 5,
        scoring: str = "accuracy",
        return_train_score: bool = False,
    ):
        """Initialize the grid search.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values.
            n_jobs: Number of parallel jobs (reserved for future implementation).
            verbose: Verbosity level.
            cv: Number of cross-validation folds.
            scoring: Metric name to optimize.
            return_train_score: Whether to also report training scores.
        """
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = cv
        self.scoring = scoring
        self.return_train_score = return_train_score
        
        # Results storage
        self.results_: Optional[Dict[str, Any]] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_ = 0.0

    def fit(
        self, 
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_fn: Optional[Callable] = None,
    ) -> "GridSearch":
        """Run grid search to find the best hyperparameters.
        
        Args:
            model: Model instance or class to tune.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (if None, use cross-validation).
            y_val: Validation labels.
            train_fn: Custom training function (optional).
        
        Returns:
            Self for method chaining.
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        
        combinations = list(itertools.product(*param_values))
        n_combinations = len(combinations)
        
        if self.verbose > 0:
            print(f"GridSearch: Evaluating {n_combinations} parameter combinations")
            print(f"Parameters: {param_names}")
        
        # Store results
        all_results = []
        
        for i, params in enumerate(combinations):
            param_dict = dict(zip(param_names, params))
            
            if self.verbose > 0:
                print(f"\n[{i+1}/{n_combinations}] Testing: {param_dict}")
            
            # Evaluate this parameter combination
            score = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, param_dict, train_fn
            )
            
            result = {
                "params": param_dict,
                "score": float(score),
            }
            
            if self.return_train_score:
                # Training score would go here
                pass
            
            all_results.append(result)
        
        # Find best parameters
        scores = [r["score"] for r in all_results]
        best_idx = np.argmax(scores)
        
        self.best_params_ = all_results[best_idx]["params"]
        self.best_score_ = float(all_results[best_idx]["score"])
        
        if self.verbose > 0:
            print(f"\nBest parameters found: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")
        
        # Store all results
        self.results_ = {
            "params": [r["params"] for r in all_results],
            "scores": scores,
            "all_results": all_results,
        }
        
        return self

    def _evaluate_model(
        self, 
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        params: Dict[str, Any],
        train_fn: Optional[Callable],
    ) -> float:
        """Evaluate a model with given parameters.
        
        Args:
            model: Model instance or class.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            params: Hyperparameter values to use.
            train_fn: Custom training function.
        
        Returns:
            Evaluation score.
        """
        # Apply parameters to model
        if hasattr(model, 'set_params'):
            model.set_params(**params)
        elif isinstance(model, dict):
            model.update(params)
        
        # Train and evaluate
        if train_fn is not None:
            return train_fn(X_train, y_train, X_val, y_val, params)
        
        # Default training approach
        try:
            from .tuner import _train_and_evaluate
            score = _train_and_evaluate(model, X_train, y_train, X_val, y_val, params)
            return score
        except Exception as e:
            if self.verbose > 0:
                print(f"Error evaluating parameters {params}: {e}")
            return float('-inf')

    def cross_validate(
        self, 
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation for each parameter combination.
        
        Args:
            model: Model instance or class to tune.
            X: All features.
            y: All labels.
            train_fn: Custom training function.
        
        Returns:
            Dictionary containing CV scores for each parameter combination.
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        
        combinations = list(itertools.product(*param_values))
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        all_cv_results = []
        
        for i, params in enumerate(combinations):
            param_dict = dict(zip(param_names, params))
            
            if self.verbose > 0:
                print(f"\n[{i+1}/{len(combinations)}] CV Testing: {param_dict}")
            
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                score = self._evaluate_model(
                    model, X_train_fold, y_train_fold, 
                    X_val_fold, y_val_fold, param_dict, train_fn
                )
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            all_cv_results.append({
                "params": param_dict,
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "fold_scores": cv_scores,
            })
        
        # Find best parameters based on CV mean score
        best_idx = np.argmax([r["mean_score"] for r in all_cv_results])
        
        self.best_params_ = all_cv_results[best_idx]["params"]
        self.best_score_ = float(all_cv_results[best_idx]["mean_score"])
        
        if self.verbose > 0:
            print(f"\nBest parameters found: {self.best_params_}")
            print(f"Best CV score: {self.best_score_:.4f} (+/- {all_cv_results[best_idx]['std_score']:.4f})")
        
        # Store results
        self.results_ = {
            "params": [r["params"] for r in all_cv_results],
            "mean_scores": [r["mean_score"] for r in all_cv_results],
            "std_scores": [r["std_score"] for r in all_cv_results],
            "all_results": all_cv_results,
        }
        
        return self.results_

    def get_params(self) -> Dict[str, List[Any]]:
        """Get the parameter grid.
        
        Returns:
            Dictionary mapping parameter names to lists of values.
        """
        return self.param_grid.copy()

    def set_params(self, param_grid: Dict[str, List[Any]]) -> None:
        """Set the parameter grid.
        
        Args:
            param_grid: New parameter grid dictionary.
        """
        self.param_grid = param_grid

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Get search results."""
        return self.results_

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found."""
        return self.best_params_

    @property
    def best_score(self) -> float:
        """Get the best score achieved."""
        return self.best_score_


class GridSearchConfig:
    """Configuration for grid search operations."""

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5,
        verbose: int = 1,
        scoring: str = "accuracy",
    ):
        """Initialize configuration.
        
        Args:
            param_grid: Parameter grid dictionary.
            cv_folds: Number of CV folds.
            verbose: Verbosity level.
            scoring: Metric to optimize.
        """
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.scoring = scoring

    def create_search(self) -> GridSearch:
        """Create a GridSearch instance with this configuration."""
        return GridSearch(
            param_grid=self.param_grid,
            verbose=self.verbose,
            cv=self.cv_folds,
            scoring=self.scoring,
        )


def generate_param_grid(
    model_type: str,
    base_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Any]]:
    """Generate a default parameter grid for common model types.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'rnn', 'transformer').
        base_params: Base parameters to use as defaults.
    
    Returns:
        Parameter grid dictionary.
    """
    default_grids = {
        "mlp": {
            "hidden_dims": [[128], [256, 128], [512, 256, 128]],
            "learning_rate": [0.001, 0.01, 0.1],
            "dropout": [0.0, 0.3, 0.5],
            "batch_size": [16, 32, 64],
        },
        "cnn": {
            "num_filters": [[32, 64], [64, 128]],
            "kernel_sizes": [[3], [3, 5]],
            "learning_rate": [0.001, 0.01],
            "dropout": [0.0, 0.3, 0.5],
        },
        "rnn": {
            "hidden_dims": [[64], [128], [64, 128]],
            "learning_rate": [0.001, 0.01],
            "dropout": [0.0, 0.3],
        },
        "transformer": {
            "d_model": [128, 256],
            "num_heads": [4, 8],
            "learning_rate": [0.0001, 0.001],
        },
    }
    
    if model_type not in default_grids:
        raise ValueError(f"Unknown model type: {model_type}")
    
    grid = default_grids[model_type].copy()
    
    if base_params:
        for key, value in base_params.items():
            if key in grid:
                grid[key] = value
    
    return grid
