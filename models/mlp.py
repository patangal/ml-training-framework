"""
Multi-Layer Perceptron (MLP) implementation for ML Framework.

This module provides a flexible MLP architecture suitable for:
- Tabular data classification/regression
- Feature extraction
- Baseline models for comparison
"""

from typing import Dict, Any, List, Optional, Tuple


class MLPConfig(ModelConfig):
    """Configuration class for MLP models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default values specific to MLPs
        defaults = {
            "input_dim": 128,
            "hidden_dims": [256, 128],
            "output_dim": 10,
            "activation": "relu",
            "dropout": 0.5,
            "use_batch_norm": True,
            "use_bias": True,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.config.get("input_dim", 128)

    @property
    def hidden_dims(self) -> List[int]:
        """Get list of hidden layer sizes."""
        return self.config.get("hidden_dims", [256, 128])

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self.config.get("output_dim", 10)

    @property
    def activation(self) -> str:
        """Get activation function name."""
        return self.config.get("activation", "relu")

    @property
    def dropout(self) -> float:
        """Get dropout rate."""
        return self.config.get("dropout", 0.5)

    @property
    def use_batch_norm(self) -> bool:
        """Check if batch normalization is used."""
        return self.config.get("use_batch_norm", True)

    @property
    def use_bias(self) -> bool:
        """Check if bias terms are used."""
        return self.config.get("use_bias", True)


class MLP(BaseModel):
    """Multi-Layer Perceptron (Fully Connected Neural Network).
    
    A classic neural network architecture consisting of fully connected layers.
    Suitable for tabular data and as a baseline model.
    
    Example usage:
        config = MLPConfig(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            output_dim=10,
            dropout=0.3,
        )
        model = MLP(config)
    """

    def __init__(self, config: Optional[MLPConfig] = None):
        super().__init__(config)
        
        self.config = config or MLPConfig()
        self._layers: List[Any] = []

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the MLP architecture.
        
        Args:
            input_shape: Shape of input data (features,).
        """
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]
        
        self.input_dim = input_dim
        
        # Create hidden layers
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            layer_name = f"fc_{i}"
            
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim, bias=self.config.use_bias)
            
            # Batch normalization (optional)
            bn = nn.BatchNorm1d(hidden_dim) if self.config.use_batch_norm else None
            
            # Activation function
            activation = self._get_activation(self.config.activation)
            
            # Dropout (optional)
            dropout_layer = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else None
            
            # Build layer sequence
            layer_seq = [linear]
            if bn is not None:
                layer_seq.append(bn)
            layer_seq.append(activation)
            if dropout_layer is not None:
                layer_seq.append(dropout_layer)
            
            self._layers.append((layer_name, nn.Sequential(*layer_seq)))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation for logits)
        output_linear = nn.Linear(prev_dim, self.config.output_dim, bias=self.config.use_bias)
        self._layers.append(("output", output_linear))

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}. Available: {list(activations.keys())}")
        
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Ensure 2D input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for _, layer in self._layers:
            x = layer(x)
        
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if not hasattr(self, 'model'):
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer."""
        info = []
        
        prev_dim = self.input_dim
        
        for i, (name, _) in enumerate(self._layers[:-1]):  # Exclude output
            hidden_dim = self.config.hidden_dims[i] if i < len(self.config.hidden_dims) else self._layers[-2][1].out_features
            
            params = prev_dim * hidden_dim + hidden_dim  # weights + bias
            if self.config.use_batch_norm:
                params += 2 * hidden_dim  # gamma and beta
            
            info.append({
                "type": f"FC_{i}",
                "params": params,
                "output_shape": f"(?, {hidden_dim})",
                "trainable": True,
            })
            
            prev_dim = hidden_dim
        
        # Output layer
        output_params = prev_dim * self.config.output_dim + self.config.output_dim
        info.append({
            "type": "Output",
            "params": output_params,
            "output_shape": f"(?, {self.config.output_dim})",
            "trainable": True,
        })
        
        return info

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        torch.save({
            "config": self.config.to_dict(),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MLP":
        """Load a model from disk."""
        checkpoint = torch.load(filepath)
        config = MLPConfig(**checkpoint["config"])
        model = cls(config)
        
        if hasattr(model, 'state_dict') and checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"])
        
        return model


# Pure PyTorch implementation

class MLPPyTorch(nn.Module):
    """Pure PyTorch MLP implementation."""

    def __init__(self, config: Optional[MLPConfig] = None):
        super().__init__()
        self.config = config or MLPConfig()
        
        layers = []
        prev_dim = self.config.input_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=self.config.use_bias))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(self.config.activation))
            
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.config.output_dim, bias=self.config.use_bias))
        
        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Ensure 2D input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TensorFlow/Keras implementation

class MLPKeras(keras.Model):
    """TensorFlow/Keras MLP implementation."""

    def __init__(self, config: Optional[MLPConfig] = None):
        super().__init__()
        self.config = config or MLPConfig()
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            if self.config.use_batch_norm:
                self.add(layers.BatchNormalization())
            
            self.add(layers.Dense(hidden_dim, activation=self._get_activation_keras(self.config.activation)))
            
            if self.config.dropout > 0:
                self.add(layers.Dropout(self.config.dropout))
        
        # Output layer
        self.output_layer = layers.Dense(
            self.config.output_dim, 
            activation=None  # No activation for logits
        )

    def _get_activation_keras(self, name: str):
        """Get Keras activation function by name."""
        activations = {
            "relu": "relu",
            "leaky_relu": "leaky_relu",
            "tanh": "tanh",
            "sigmoid": "sigmoid",
            "gelu": "gelu",
            "elu": "elu",
            "selu": "selu",
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        # Ensure 2D input
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
        
        x = inputs
        
        for layer in self.layers[:-1]:  # All except output
            x = layer(x, training=training)
        
        return self.output_layer(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(v.shape.num_elements() for v in self.trainable_variables)


# Register MLP models
ModelFactory.register("mlp_pytorch")(MLPPyTorch)
ModelFactory.register("mlp_keras")(MLPKeras)
