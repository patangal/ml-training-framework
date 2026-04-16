"""
Bayesian Optimization hyperparameter tuning for ML Framework.

This module provides Bayesian optimization using Gaussian processes,
which is more sample-efficient than random or grid search for expensive evaluations.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize


class BayesianOptimization:
    """Bayesian optimization for hyperparameter tuning.
    
    Uses a probabilistic model (Gaussian process) to approximate the objective function
    and an acquisition function to decide which parameters to evaluate next.
    
    Args:
        param_bounds: Dictionary mapping parameter names to bounds.
            Bounds can be tuples (low, high) for continuous or lists for discrete.
        n_iter: Number of iterations to run.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level.
        alpha: Noise variance for GP (regularization).
    """

    def __init__(
        self, 
        param_bounds: Dict[str, Tuple[float, float]],
        n_iter: int = 20,
        random_state: Optional[int] = None,
        verbose: int = 1,
        alpha: float = 1e-6,
    ):
        """Initialize the Bayesian optimizer.
        
        Args:
            param_bounds: Dictionary of parameter bounds.
            n_iter: Number of iterations to run.
            random_state: Random seed for reproducibility.
            verbose: Verbosity level.
            alpha: Noise variance for GP.
        """
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        
        # Results storage
        self.results_: List[Dict[str, Any]] = []
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_ = float('-inf')
        
        # GP state
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(
        self, 
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_fn: Optional[Callable] = None,
    ) -> "BayesianOptimization":
        """Run Bayesian optimization to find the best hyperparameters.
        
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
        
        # Initialize GP state
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)
        
        # Initial random samples (warm-up)
        n_warmup = min(5, self.n_iter // 2)
        X_init = np.zeros((n_warmup, n_params))
        
        for i in range(n_warmup):
            X_init[i] = [self._sample_param(name) for name in param_names]
        
        # Evaluate initial samples
        y_init = []
        for i in range(n_warmup):
            params = dict(zip(param_names, X_init[i]))
            
            if self.verbose > 0:
                print(f"\n[Warm-up {i+1}/{n_warmup}] Testing: {params}")
            
            score = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, params, train_fn
            )
            y_init.append(score)
        
        # Store initial results
        for i in range(n_warmup):
            self.results_.append({
                "params": dict(zip(param_names, X_init[i])),
                "score": float(y_init[i]),
            })
        
        self.X_train = X_init.copy()
        self.y_train = np.array(y_init)
        
        # Main optimization loop
        for i in range(n_warmup, self.n_iter):
            if self.verbose > 0:
                print(f"\n[Iteration {i+1}/{self.n_iter}]")
            
            # Fit GP to current data
            gp = GaussianProcess(self.X_train, self.y_train, alpha=self.alpha)
            
            # Find next point using acquisition function (EI)
            X_next = self._maximize_acquisition(gp, param_names)
            
            # Evaluate at this point
            params = dict(zip(param_names, X_next))
            
            if self.verbose > 0:
                print(f"Testing: {params}")
            
            score = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, params, train_fn
            )
            
            # Store result
            self.results_.append({
                "params": params,
                "score": float(score),
            })
            
            # Update GP training data
            self.X_train = np.vstack([self.X_train, X_next.reshape(1, -1)])
            self.y_train = np.append(self.y_train, score)
        
        # Find best parameters
        scores = [r["score"] for r in self.results_]
        best_idx = np.argmax(scores)
        
        self.best_params_ = self.results_[best_idx]["params"]
        self.best_score_ = float(self.results_[best_idx]["score"])
        
        if self.verbose > 0:
            print(f"\nBest parameters found: {self.best_params_}")
            print(f"Best score: {self.best_score_:.4f}")
        
        return self

    def _sample_param(self, param_name: str) -> float:
        """Sample a parameter value from its bounds.
        
        Args:
            param_name: Name of the parameter.
        
        Returns:
            Sampled value within bounds.
        """
        low, high = self.param_bounds[param_name]
        return np.random.uniform(low, high)

    def _maximize_acquisition(
        self, 
        gp: "GaussianProcess", 
        param_names: List[str]
    ) -> np.ndarray:
        """Maximize the acquisition function to find next evaluation point.
        
        Args:
            gp: Fitted Gaussian process model.
            param_names: List of parameter names.
        
        Returns:
            Next point to evaluate.
        """
        # Bounds for optimization (normalized)
        bounds = []
        for name in param_names:
            low, high = self.param_bounds[name]
            # Normalize to [0, 1]
            norm_low = (low - gp.X_train.min(axis=0)[param_names.index(name)]) / \
                       (gp.X_train.max(axis=0)[param_names.index(name)] - gp.X_train.min(axis=0)[param_names.index(name)]) + 1e-6
            norm_high = 1.0 - 1e-6
            bounds.append((norm_low, norm_high))
        
        # Maximize expected improvement
        result = minimize(
            lambda x: -self._expected_improvement(gp, x),
            x0=np.ones(len(param_names)),
            method='Nelder-Mead',
            options={'maxiter': 50}
        )
        
        return result.x

    def _expected_improvement(self, gp: "GaussianProcess", X: np.ndarray) -> float:
        """Compute expected improvement at a point.
        
        Args:
            gp: Fitted Gaussian process model.
            X: Point to evaluate EI at.
        
        Returns:
            Expected improvement value (negative for minimization).
        """
        # Predict mean and variance at X
        mu, var = gp.predict(X.reshape(1, -1))
        sigma = np.sqrt(var)
        
        # Current best
        y_best = self.y_train.max() if self.best_score_ == float('-inf') else self.best_score_
        
        # Standardized expected improvement
        Z = (y_best - mu[0]) / (sigma + 1e-6)
        
        ei = (y_best - mu[0]) * Z * np.exp(-0.5 * Z**2) / sigma + \
             sigma * Z * np.exp(-0.5 * Z**2) if sigma > 0 else 0
        
        return ei

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
            Evaluation score (higher is better).
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

    @property
    def results(self) -> List[Dict[str, Any]]:
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


class GaussianProcess:
    """Simple Gaussian Process implementation for Bayesian optimization.
    
    This is a lightweight GP implementation suitable for low-dimensional
    hyperparameter spaces in Bayesian optimization.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, alpha: float = 1e-6):
        """Initialize the Gaussian Process.
        
        Args:
            X: Training input data (n_samples, n_features).
            y: Training target values (n_samples,).
            alpha: Noise variance for regularization.
        """
        self.X_train = X
        self.y_train = y
        self.alpha = alpha
        
        # Compute and store kernel matrix
        self.K = self._compute_kernel(X, X) + alpha * np.eye(len(y))
        
        # Solve K^-1 @ y for efficient prediction
        try:
            self.L = np.linalg.cholesky(self.K)
            self.alpha_coeffs = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        except np.linalg.LinAlgError:
            # Fallback to direct solve if Cholesky fails
            self.alpha_coeffs = np.linalg.solve(self.K, y)

    def _compute_kernel(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray, 
        length_scale: float = 1.0
    ) -> np.ndarray:
        """Compute RBF (squared exponential) kernel matrix.
        
        Args:
            X1: First input matrix.
            X2: Second input matrix.
            length_scale: Length scale parameter for the kernel.
        
        Returns:
            Kernel matrix K(X1, X2).
        """
        # Compute pairwise distances
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                  np.sum(X2**2, axis=1) - \
                  2 * np.dot(X1, X2.T)
        
        # Ensure non-negative (numerical stability)
        sq_dist = np.maximum(sq_dist, 0)
        
        # RBF kernel
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2))
        
        return K

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points.
        
        Args:
            X: Points to predict at (n_samples, n_features).
        
        Returns:
            Tuple of (mean predictions, variance predictions).
        """
        # Compute kernel between training and test points
        K_star = self._compute_kernel(self.X_train, X)
        K_star_star = self._compute_kernel(X, X)
        
        # Predict mean
        mu = np.dot(K_star.T, self.alpha_coeffs)
        
        # Predict variance
        v = np.linalg.solve(self.L, K_star)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        
        # Ensure non-negative variance (numerical stability)
        var = np.maximum(var, 1e-6)
        
        return mu, var


class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iter: int = 20,
        verbose: int = 1,
        random_state: Optional[int] = None,
    ):
        """Initialize configuration.
        
        Args:
            param_bounds: Parameter bounds dictionary.
            n_iter: Number of iterations to run.
            verbose: Verbosity level.
            random_state: Random seed for reproducibility.
        """
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def create_optimizer(self) -> BayesianOptimization:
        """Create a BayesianOptimization instance with this configuration."""
        return BayesianOptization(
            param_bounds=self.param_bounds,
            n_iter=self.n_iter,
            verbose=self.verbose,
            random_state=self.random_state,
        )


def generate_param_bounds(
    model_type: str,
) -> Dict[str, Tuple[float, float]]:
    """Generate default parameter bounds for common model types.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'rnn', 'transformer').
    
    Returns:
        Parameter bounds dictionary with log-scaled ranges.
    """
    default_bounds = {
        "mlp": {
            "learning_rate": (1e-5, 1e-1),
            "dropout": (0.0, 0.7),
            "batch_size": (8, 256),
        },
        "cnn": {
            "learning_rate": (1e-5, 1e-2),
            "dropout": (0.0, 0.7),
            "num_filters": (16, 256),
        },
        "rnn": {
            "learning_rate": (1e-5, 1e-2),
            "dropout": (0.0, 0.7),
            "hidden_dims": (32, 512),
        },
        "transformer": {
            "learning_rate": (1e-6, 1e-4),
            "d_model": (64, 512),
            "num_heads": (2, 16),
        },
    }
    
    if model_type not in default_bounds:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return default_bounds[model_type]
