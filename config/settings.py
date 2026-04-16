"""
Configuration management utilities for ML Framework.

This module provides configuration classes that can be used to:
- Define training hyperparameters
- Specify model architecture settings
- Configure backend options (PyTorch/TensorFlow)
- Load/save configurations from files
"""

from typing import Dict, Any, List, Optional, Union
import json
import os


class Config:
    """Base configuration class for ML Framework.
    
    Provides a unified interface for managing all framework configurations.
    Supports loading/saving to JSON/YAML files and merging configurations.
    
    Args:
        **kwargs: Initial configuration parameters.
    """

    def __init__(self, **kwargs):
        """Initialize the configuration."""
        self._config: Dict[str, Any] = {}
        
        for key, value in kwargs.items():
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys).
        
        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Key '{key}' not found")
        
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys).
            value: Value to set.
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def __delitem__(self, key: str) -> None:
        """Delete a configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys).
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                raise KeyError(f"Key '{key}' not found")
        
        del config[keys[-1]]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default.
        
        Args:
            key: Configuration key.
            default: Default value if key not found.
        
        Returns:
            Configuration value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        """Get all top-level configuration keys."""
        return list(self._config.keys())

    def items(self) -> List[tuple]:
        """Get all configuration key-value pairs."""
        return list(self._config.items())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return self._config.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
        
        Returns:
            A new Config instance.
        """
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file.
        
        Returns:
            A Config instance loaded from the file.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def merge(self, other: "Config") -> None:
        """Merge another configuration into this one.
        
        Args:
            other: Configuration to merge from.
        """
        self._merge_dicts(self._config, other.to_dict())

    def _merge_dicts(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) and 
                isinstance(value, dict)):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._config, indent=2)})"


class TrainingConfig(Config):
    """Configuration for training parameters.
    
    Contains settings related to the training process including:
    - Number of epochs and batch size
    - Learning rate and optimizer settings
    - Early stopping and callbacks
    - Device preferences (CPU/GPU)
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Default values
        defaults = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "device": "auto",
            "mixed_precision": False,
            
            # Early stopping
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
            "early_stopping_monitor": "val_loss",
            
            # Learning rate scheduling
            "lr_scheduler_type": None,
            "lr_scheduler_warmup_epochs": 5,
            "lr_scheduler_decay_rate": 0.95,
            
            # Callbacks
            "save_checkpoints": True,
            "checkpoint_dir": "checkpoints",
            "checkpoint_every_n_epochs": 1,
            "tensorboard_log_dir": "logs",
        }
        
        for key, value in defaults.items():
            self[key] = value
        
        # Override with provided values
        for key, value in kwargs.items():
            self[key] = value

    @property
    def epochs(self) -> int:
        """Get number of training epochs."""
        return self.get("epochs", 100)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.get("batch_size", 32)

    @property
    def learning_rate(self) -> float:
        """Get initial learning rate."""
        return self.get("learning_rate", 0.001)

    @property
    def optimizer(self) -> str:
        """Get optimizer type."""
        return self.get("optimizer", "adam")

    @property
    def device(self) -> str:
        """Get device preference."""
        return self.get("device", "auto")


class ModelConfig(Config):
    """Configuration for model architecture.
    
    Contains settings related to the neural network model including:
    - Model type (CNN, RNN, Transformer, MLP)
    - Architecture hyperparameters
    - Regularization settings
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Default values
        defaults = {
            "model_type": "mlp",
            "input_dim": 128,
            "output_dim": 10,
            
            # Common settings
            "activation": "relu",
            "dropout": 0.5,
            "batch_norm": True,
            
            # MLP-specific
            "hidden_dims": [256, 128],
            
            # CNN-specific
            "num_filters": [32, 64, 128],
            "kernel_sizes": [3],
            "pool_sizes": [2],
            
            # RNN-specific
            "rnn_type": "lstm",
            "bidirectional": False,
            
            # Transformer-specific
            "num_heads": 8,
            "d_model": 512,
            "d_ff": 2048,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
        }
        
        for key, value in defaults.items():
            self[key] = value
        
        # Override with provided values
        for key, value in kwargs.items():
            self[key] = value

    @property
    def model_type(self) -> str:
        """Get model type."""
        return self.get("model_type", "mlp")


class BackendConfig(Config):
    """Configuration for backend settings.
    
    Contains settings related to the deep learning framework including:
    - Framework selection (PyTorch/TensorFlow)
    - Device configuration
    - Mixed precision training
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Default values
        defaults = {
            "framework": "pytorch",  # or "tensorflow"
            "device": "auto",  # cpu, cuda, auto
            "mixed_precision": False,
            
            # PyTorch-specific
            "num_workers": 0,
            "pin_memory": True,
            
            # TensorFlow-specific
            "allow_growth": True,
        }
        
        for key, value in defaults.items():
            self[key] = value
        
        # Override with provided values
        for key, value in kwargs.items():
            self[key] = value

    @property
    def framework(self) -> str:
        """Get selected framework."""
        return self.get("framework", "pytorch")


class EvaluationConfig(Config):
    """Configuration for evaluation settings.
    
    Contains settings related to model evaluation including:
    - Metrics to compute
    - Batch size for evaluation
    - Report format
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Default values
        defaults = {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "batch_size": 32,
            "verbose": True,
            
            # Classification report settings
            "classification_report": True,
            
            # Regression report settings
            "regression_report": False,
        }
        
        for key, value in defaults.items():
            self[key] = value
        
        # Override with provided values
        for key, value in kwargs.items():
            self[key] = value

    @property
    def metrics(self) -> List[str]:
        """Get list of metric names to compute."""
        return self.get("metrics", ["accuracy"])


class TuningConfig(Config):
    """Configuration for hyperparameter tuning.
    
    Contains settings related to hyperparameter optimization including:
    - Search method (grid, random, bayesian)
    - Parameter bounds/distributions
    - Number of iterations
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Default values
        defaults = {
            "enabled": False,
            "method": "random",  # grid, random, bayesian
            "n_iter": 10,
            "cv_folds": 5,
            
            # Parameter bounds (for Bayesian optimization)
            "param_bounds": {},
        }
        
        for key, value in defaults.items():
            self[key] = value
        
        # Override with provided values
        for key, value in kwargs.items():
            self[key] = value

    @property
    def enabled(self) -> bool:
        """Check if tuning is enabled."""
        return self.get("enabled", False)


class ConfigManager:
    """Manages configuration loading and merging.
    
    Provides utilities for:
    - Loading configurations from files
    - Merging multiple configuration sources
    - Validating configurations
    """

    @staticmethod
    def load_from_file(filepath: str, config_type: type = Config) -> Config:
        """Load a configuration from a file.
        
        Args:
            filepath: Path to the configuration file.
            config_type: Type of configuration class to instantiate.
        
        Returns:
            Loaded configuration instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        return config_type.load(filepath)

    @staticmethod
    def merge_configs(*configs: Config) -> Config:
        """Merge multiple configurations into one.
        
        Later configurations override earlier ones.
        
        Args:
            *configs: Configuration instances to merge.
        
        Returns:
            Merged configuration instance.
        """
        if not configs:
            return Config()
        
        merged = configs[0]
        for config in configs[1:]:
            merged.merge(config)
        
        return merged

    @staticmethod
    def validate(config: Config, required_keys: List[str]) -> bool:
        """Validate that a configuration has all required keys.
        
        Args:
            config: Configuration to validate.
            required_keys: List of required key names.
        
        Returns:
            True if all required keys are present.
        """
        for key in required_keys:
            try:
                config[key]
            except KeyError:
                return False
        
        return True

    @staticmethod
    def get_defaults(config_type: type) -> Config:
        """Get default configuration for a specific type.
        
        Args:
            config_type: Configuration class type.
        
        Returns:
            Default configuration instance.
        """
        if config_type == TrainingConfig:
            return TrainingConfig()
        elif config_type == ModelConfig:
            return ModelConfig()
        elif config_type == BackendConfig:
            return BackendConfig()
        elif config_type == EvaluationConfig:
            return EvaluationConfig()
        elif config_type == TuningConfig:
            return TuningConfig()
        else:
            return Config()


# Convenience functions

def create_training_config(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "auto",
) -> TrainingConfig:
    """Create a training configuration with common defaults.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        device: Device preference (cpu, cuda, auto).
    
    Returns:
        Configured TrainingConfig instance.
    """
    return TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )


def create_model_config(
    model_type: str = "mlp",
    input_dim: int = 128,
    output_dim: int = 10,
) -> ModelConfig:
    """Create a model configuration with common defaults.
    
    Args:
        model_type: Type of model (mlp, cnn, rnn, transformer).
        input_dim: Input dimension.
        output_dim: Output dimension.
    
    Returns:
        Configured ModelConfig instance.
    """
    return ModelConfig(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def create_backend_config(
    framework: str = "pytorch",
    device: str = "auto",
) -> BackendConfig:
    """Create a backend configuration with common defaults.
    
    Args:
        framework: Deep learning framework (pytorch, tensorflow).
        device: Device preference.
    
    Returns:
        Configured BackendConfig instance.
    """
    return BackendConfig(
        framework=framework,
        device=device,
    )
