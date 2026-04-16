"""
Base model classes and configuration for ML Framework.

This module provides abstract base classes and configuration utilities for defining
neural network models in a framework-agnostic way.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import os


class ModelConfig:
    """Configuration class for neural network models.
    
    Provides a structured way to define model hyperparameters and settings.
    Supports serialization/deserialization for saving/loading configurations.
    
    Args:
        **kwargs: Configuration parameters specific to the model type.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @property
    def input_dim(self) -> Optional[int]:
        """Get input dimension."""
        return self.config.get("input_dim") or self.config.get("input_shape", [None])[0] if isinstance(self.config.get("input_shape"), list) else None

    @property
    def output_dim(self) -> Optional[int]:
        """Get output dimension."""
        return self.config.get("output_dim")

    @property
    def hidden_dims(self) -> List[int]:
        """Get hidden layer dimensions."""
        return self.config.get("hidden_dims", []) or self.config.get("layers", [])

    @property
    def activation(self) -> str:
        """Get activation function name."""
        return self.config.get("activation", "relu")

    @property
    def dropout_rate(self) -> float:
        """Get dropout rate."""
        return self.config.get("dropout", 0.0) or self.config.get("dropout_rate", 0.0)

    @property
    def batch_norm(self) -> bool:
        """Check if batch normalization is enabled."""
        return self.config.get("batch_norm", False)

    @property
    def use_bias(self) -> bool:
        """Check if bias terms are used."""
        return self.config.get("use_bias", True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key.
            default: Default value if key not found.
        
        Returns:
            Configuration value or default.
        """
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
        
        Returns:
            A new ModelConfig instance.
        """
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ModelConfig":
        """Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file.
        
        Returns:
            A ModelConfig instance loaded from the file.
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        return f"ModelConfig({self.config})"


class BaseModel(ABC):
    """Abstract base class for neural network models.
    
    This class defines the interface that all model implementations must follow,
    ensuring consistency across different architectures and backends.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the base model.
        
        Args:
            config: Model configuration object.
        """
        self.config = config or ModelConfig()
        self._built = False

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the model architecture.
        
        Args:
            input_shape: Shape of the input data (excluding batch dimension).
        """
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.
        
        Args:
            x: Input tensor/data.
        
        Returns:
            Model output/predictions.
        """
        pass

    def __call__(self, x: Any) -> Any:
        """Allow calling the model directly."""
        return self.forward(x)

    @abstractmethod
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters.
        """
        pass

    @abstractmethod
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer in the model.
        
        Returns:
            List of dictionaries containing layer information.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path where the model will be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Loaded model instance.
        """
        pass

    def summary(self) -> None:
        """Print a summary of the model architecture."""
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)
        print(f"Model Class: {self.__class__.__name__}")
        
        layers_info = self.get_layer_info()
        total_params = 0
        
        for i, layer in enumerate(layers_info):
            params = layer.get("params", 0)
            total_params += params
            print(f"{i+1}. {layer['type']:20s} | Output: {str(layer['output_shape']):25s} | Params: {params:>10,}")
        
        print("-" * 60)
        print(f"Total Parameters: {total_params:,}")
        trainable = sum(l.get("params", 0) for l in layers_info if not l.get("trainable", True))
        print(f"Trainable Parameters: {total_params - trainable:,}")
        print("=" * 60)

    def freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specified layers (set their gradients to zero).
        
        Args:
            layer_names: Names of layers to freeze.
        """
        pass

    def unfreeze_all_layers(self) -> None:
        """Unfreeze all previously frozen layers."""
        pass


class ModelFactory:
    """Factory class for creating model instances.
    
    Provides a centralized way to create models based on configuration or type.
    """

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class.
        
        Args:
            name: The name to register the model under.
        
        Returns:
            Decorator function.
        """
        def decorator(klass):
            cls._models[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, model_type: str, config: Optional[ModelConfig] = None, **kwargs) -> BaseModel:
        """Create a model instance by type.
        
        Args:
            model_type: Type of model to create ('cnn', 'rnn', 'transformer', 'mlp').
            config: Model configuration object.
            **kwargs: Additional arguments for the model constructor.
        
        Returns:
            A BaseModel instance.
        
        Raises:
            ValueError: If unknown model type is specified.
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}")
        
        return cls._models[model_type](config, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model types.
        
        Returns:
            List of available model type names.
        """
        return list(cls._models.keys())


# Register built-in models
from .cnn import CNN
from .rnn import RNN, LSTM, GRU
from .transformer import Transformer
from .mlp import MLP

ModelFactory.register("cnn")(CNN)
ModelFactory.register("rnn")(RNN)
ModelFactory.register("lstm")(LSTM)
ModelFactory.register("gru")(GRU)
ModelFactory.register("transformer")(Transformer)
ModelFactory.register("mlp")(MLP)


def get_model(model_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseModel:
    """Convenience function to create a model by type.
    
    Args:
        model_type: Type of model ('cnn', 'rnn', 'transformer', 'mlp').
        config: Configuration dictionary or ModelConfig object.
        **kwargs: Additional arguments for the model constructor.
    
    Returns:
        A BaseModel instance.
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    return ModelFactory.create(model_type, config=config, **kwargs)
