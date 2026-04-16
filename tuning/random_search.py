"""
Random Search hyperparameter optimization for ML Framework.

This module provides random sampling of hyperparameters from specified distributions,
which can be more efficient than grid search for high-dimensional spaces.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import itertools


class RandomSearch:
    """Random search for hyperparameter optimization.
    
    Samples random combinations of hyperparameters from specified distributions.
    Often more efficient than grid search, especially when some parameters have
    little effect on performance.
    
    Args:
        param_distributions: Dictionary mapping parameter names to distributions.
            Distributions can be:
            - List: Sample uniformly from values
            - Tuple (low, high): Sample uniformly in range
            - Callable: Call to generate a value
        n_iter: Number of parameter settings to sample.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level.
        cv: Cross-validation fold count.
    """

    def __init__(
        self, 
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        random_state: Optional[int] = None,
        verbose: int = 1,
        cv: int = 5,
        scoring: str = "accuracy",
    ):
        """Initialize the random search.
        
        Args:
            param_distributions: Dictionary of parameter names to distributions.
            n_iter: Number of random combinations to try.
            random_state: Random seed for reproducibility.
            verbose: Verbosity level.
            cv: Number of cross-validation folds.
            scoring: Metric name to optimize.
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.cv = cv
        self.scoring = scoring
        
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
    ) -> "RandomSearch":
        """Run random search to find the best hyperparameters.
        
        Args:
            model: Model instance or class to tune.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (if None, use cross-validation).
            y_val: Validation labels.
            train_fn: Custom training function.
        
        Returns:
            Self for method chaining.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.verbose > 0:
            print(f"RandomSearch: Evaluating {self.n_iter} random parameter combinations")
        
        # Store results
        all_results = []
        
        for i in range(self.n_iter):
            # Sample parameters from distributions
            params = self._sample_parameters()
            
            if self.verbose > 0:
                print(f"\n[{i+1}/{self.n_iter}] Testing: {params}")
            
            # Evaluate this parameter combination
            score = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, params, train_fn
            )
            
            result = {
                "params": params,
                "score": float(score),
            }
            
            all_results.append(result)
        
        # Find best parameters
        scores = [r["score"] for r in all_results]
        best_idx = np.argmax(scores)
        
        self.best_params_ = all_results[best_idx]["params"]
        self.best_score_ = float(all_results[best_idx]["score"])
        
        if self.verbose > 0:
            print(f"\nBest parameters found: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")
        
        # Store results sorted by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        self.results_ = {
            "params": [r["params"] for r in all_results],
            "scores": scores,
            "all_results": all_results,
        }
        
        return self

    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample a set of parameters from the distributions.
        
        Returns:
            Dictionary of sampled parameter values.
        """
        params = {}
        
        for param_name, distribution in self.param_distributions.items():
            value = self._sample_from_distribution(distribution)
            params[param_name] = value
        
        return params

    def _sample_from_distribution(self, distribution: Any) -> Any:
        """Sample a single parameter value from its distribution.
        
        Args:
            distribution: Distribution specification (list, tuple, or callable).
        
        Returns:
            Sampled value.
        """
        if isinstance(distribution, list):
            # Uniform sampling from discrete values
            return np.random.choice(distribution)
        
        elif isinstance(distribution, tuple) and len(distribution) == 2:
            low, high = distribution
            
            # Check if log-uniform (for learning rates, etc.)
            if hasattr(self.param_distributions.get('log_uniform'), param_name):
                return np.random.lognormal(np.log(low), np.log(high/low))
            
            # Uniform sampling from range
            return np.random.uniform(low, high)
        
        elif callable(distribution):
            # Call the function to generate a value
            return distribution()
        
        else:
            # Return as-is (fixed value)
            return distribution

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
        """Perform cross-validation for random parameter combinations.
        
        Args:
            model: Model instance or class to tune.
            X: All features.
            y: All labels.
            train_fn: Custom training function.
        
        Returns:
            Dictionary containing CV scores for each parameter combination.
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        all_cv_results = []
        
        for i in range(self.n_iter):
            # Sample parameters
            params = self._sample_parameters()
            
            if self.verbose > 0:
                print(f"\n[{i+1}/{self.n_iter}] CV Testing: {params}")
            
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                score = self._evaluate_model(
                    model, X_train_fold, y_train_fold, 
                    X_val_fold, y_val_fold, params, train_fn
                )
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            all_cv_results.append({
                "params": params,
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
        
        # Store results sorted by mean score
        all_cv_results.sort(key=lambda x: x["mean_score"], reverse=True)
        
        self.results_ = {
            "params": [r["params"] for r in all_cv_results],
            "mean_scores": [r["mean_score"] for r in all_cv_results],
            "std_scores": [r["std_score"] for r in all_cv_results],
            "all_results": all_cv_results,
        }
        
        return self.results_

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


class RandomSearchConfig:
    """Configuration for random search operations."""

    def __init__(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        cv_folds: int = 5,
        verbose: int = 1,
        scoring: str = "accuracy",
        random_state: Optional[int] = None,
    ):
        """Initialize configuration.
        
        Args:
            param_distributions: Parameter distributions dictionary.
            n_iter: Number of iterations to run.
            cv_folds: Number of CV folds.
            verbose: Verbosity level.
            scoring: Metric to optimize.
            random_state: Random seed for reproducibility.
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.scoring = scoring
        self.random_state = random_state

    def create_search(self) -> RandomSearch:
        """Create a RandomSearch instance with this configuration."""
        return RandomSearch(
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            verbose=self.verbose,
            cv=self.cv_folds,
            scoring=self.scoring,
            random_state=self.random_state,
        )


# Common distribution helpers

def log_uniform(low: float, high: float):
    """Create a log-uniform distribution for parameters like learning rate.
    
    Args:
        low: Lower bound (exclusive).
        high: Upper bound (inclusive).
    
    Returns:
        A callable that samples from the log-uniform distribution.
    """
    def sample():
        return np.random.uniform(np.log(low), np.log(high))
    
    sample.log_uniform = True
    return sample


def uniform(low: float, high: float):
    """Create a uniform distribution for continuous parameters.
    
    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
    
    Returns:
        A callable that samples from the uniform distribution.
    """
    def sample():
        return np.random.uniform(low, high)
    
    return sample


def choice(values: List[Any]):
    """Create a categorical distribution for discrete parameters.
    
    Args:
        values: List of possible values to choose from.
    
    Returns:
        A callable that randomly selects from the values.
    """
    def sample():
        return np.random.choice(values)
    
    return sample


def generate_param_distributions(
    model_type: str,
    base_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate default parameter distributions for common model types.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'rnn', 'transformer').
        base_params: Base parameters to use as defaults.
    
    Returns:
        Parameter distributions dictionary.
    """
    default_distributions = {
        "mlp": {
            "hidden_dims": [[128], [256, 128], [512, 256]],
            "learning_rate": log_uniform(0.0001, 0.1),
            "dropout": choice([0.0, 0.3, 0.5]),
            "batch_size": choice([16, 32, 64, 128]),
        },
        "cnn": {
            "num_filters": [[32, 64], [64, 128]],
            "kernel_sizes": [[3], [5], [3, 5]],
            "learning_rate": log_uniform(0.0001, 0.01),
            "dropout": choice([0.0, 0.3, 0.5]),
        },
        "rnn": {
            "hidden_dims": [[64], [128], [256]],
            "learning_rate": log_uniform(0.0001, 0.01),
            "dropout": choice([0.0, 0.3]),
        },
        "transformer": {
            "d_model": choice([128, 256, 512]),
            "num_heads": choice([4, 8, 16]),
            "learning_rate": log_uniform(0.00001, 0.001),
        },
    }
    
    if model_type not in default_distributions:
        raise ValueError(f"Unknown model type: {model_type}")
    
    distributions = default_distributions[model_type].copy()
    
    if base_params:
        for key, value in base_params.items():
            if key in distributions:
                distributions[key] = value
    
    return distributions
