"""
Convolutional Neural Network (CNN) implementation for ML Framework.

This module provides CNN architectures suitable for image classification,
object detection, and other computer vision tasks.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class CNNConfig(ModelConfig):
    """Configuration class for CNN models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default values specific to CNNs
        defaults = {
            "num_classes": 10,
            "input_shape": (32, 32, 3),  # CIFAR-10 default
            "kernel_sizes": [3],
            "num_filters": [32, 64, 128],
            "pool_sizes": [2],
            "dropout_rates": [0.5],
            "use_batch_norm": True,
            "use_skip_connections": False,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        return self.config.get("num_classes", 10)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get input shape (height, width, channels)."""
        return tuple(self.config.get("input_shape", (32, 32, 3)))

    @property
    def kernel_sizes(self) -> List[int]:
        """Get list of kernel sizes for each conv layer."""
        return self.config.get("kernel_sizes", [3])

    @property
    def num_filters(self) -> List[int]:
        """Get list of filter counts for each conv block."""
        return self.config.get("num_filters", [32, 64, 128])

    @property
    def pool_sizes(self) -> List[int]:
        """Get list of pooling sizes."""
        return self.config.get("pool_sizes", [2])

    @property
    def dropout_rates(self) -> List[float]:
        """Get list of dropout rates after each conv block."""
        return self.config.get("dropout_rates", [0.5])

    @property
    def use_batch_norm(self) -> bool:
        """Check if batch normalization is used."""
        return self.config.get("use_batch_norm", True)

    @property
    def use_skip_connections(self) -> bool:
        """Check if skip connections (ResNet-style) are used."""
        return self.config.get("use_skip_connections", False)


class CNN(BaseModel):
    """Convolutional Neural Network for image classification.
    
    This implementation supports:
    - Multiple convolution blocks with increasing filter counts
    - Configurable kernel sizes and pooling operations
    - Batch normalization for faster convergence
    - Dropout for regularization
    - Skip connections (ResNet-style) for deeper networks
    
    Example usage:
        config = CNNConfig(
            input_shape=(32, 32, 3),
            num_classes=10,
            num_filters=[32, 64, 128],
            use_batch_norm=True,
        )
        model = CNN(config)
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__(config)
        
        self.config = config or CNNConfig()
        self._layers: List[Any] = []
        self._skip_connections: List[Any] = []

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the CNN architecture.
        
        Args:
            input_shape: Shape of input data (height, width, channels).
        """
        if len(input_shape) == 2:
            # Assume grayscale image
            input_shape = (*input_shape, 1)
        
        self.input_shape = input_shape
        
        # Initialize layers based on configuration
        num_blocks = len(self.config.num_filters)
        
        for i in range(num_blocks):
            block_name = f"conv_block_{i}"
            
            # Create convolutional layer(s)
            conv_layers = []
            kernel_size = self.config.kernel_sizes[i % len(self.config.kernel_sizes)]
            num_filters = self.config.num_filters[i]
            dropout_rate = self.config.dropout_rates[i % len(self.config.dropout_rates)]
            pool_size = self.config.pool_sizes[i % len(self.config.pool_sizes)]
            
            # First convolution (or residual connection)
            if i == 0:
                in_channels = input_shape[2]
            else:
                prev_filters = self.config.num_filters[i - 1]
                in_channels = prev_filters
            
            conv_layers.append(("conv", nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding="same")))
            
            if self.config.use_batch_norm:
                conv_layers.append(("bn", nn.BatchNorm2d(num_filters)))
            
            conv_layers.append(("relu", nn.ReLU()))
            
            # Pooling layer
            conv_layers.append(("pool", nn.MaxPool2d(pool_size=pool_size)))
            
            # Dropout
            if dropout_rate > 0:
                conv_layers.append(("dropout", nn.Dropout(dropout_rate)))
            
            self._layers.append((block_name, nn.Sequential(*[layer for _, layer in conv_layers])))
            
            # Skip connection (ResNet-style)
            if self.config.use_skip_connections and i < num_blocks - 1:
                skip_conv = nn.Conv2d(in_channels, num_filters, kernel_size=1)
                self._skip_connections.append(skip_conv)

        # Flatten layer
        self._layers.append(("flatten", nn.Flatten()))
        
        # Calculate input size after conv blocks
        temp_input = torch.zeros(1, *input_shape)
        for _, layer in self._layers[:-2]:  # Exclude flatten and fc layers
            temp_input = layer(temp_input)
        
        flattened_size = temp_input.shape[1]
        
        # Fully connected layers
        hidden_dims = self.config.get("fc_layers", [256, 128])
        fc_layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                fc_layers.append(("fc_in", nn.Linear(flattened_size, hidden_dim)))
            else:
                fc_layers.append(("fc", nn.Linear(self._layers[-1].out_features if hasattr(self._layers[-1], 'out_features') else flattened_size, hidden_dim)))
            
            fc_layers.append(("bn_fc", nn.BatchNorm1d(hidden_dim) if self.config.use_batch_norm else None))
            fc_layers.append(("relu_fc", nn.ReLU()))
            fc_layers.append(("dropout_fc", nn.Dropout(self.config.dropout_rate)))
        
        # Output layer
        fc_layers.append(("fc_out", nn.Linear(hidden_dims[-1], self.config.num_classes)))
        
        self._layers.append(("fc", nn.Sequential(*[layer for _, layer in fc_layers if layer is not None])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels).
        
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Ensure input is in correct format (NCHW for PyTorch)
        if x.dim() == 4 and x.shape[1] <= 4:  # Likely NHWC format
            x = x.permute(0, 3, 1, 2)
        
        skip_idx = 0
        
        for block_name, layer in self._layers[:-1]:  # Exclude final FC layer
            if "conv_block" in block_name and self.config.use_skip_connections:
                # Apply skip connection
                x_orig = x
                x = layer(x)
                
                # Apply residual connection
                if skip_idx < len(self._skip_connections):
                    skip_conv = self._skip_connections[skip_idx]
                    x = x + skip_conv(x_orig)
                
                x = nn.functional.relu(x)
                skip_idx += 1
            else:
                x = layer(x)
        
        # Final FC layers
        for _, layer in self._layers[-1]:
            x = layer(x)
        
        return x

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        if not hasattr(self, 'model'):
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer in the model."""
        info = []
        
        # Convolutional blocks
        num_blocks = len(self.config.num_filters)
        for i in range(num_blocks):
            kernel_size = self.config.kernel_sizes[i % len(self.config.kernel_sizes)]
            pool_size = self.config.pool_sizes[i % len(self.config.pool_sizes)]
            
            info.append({
                "type": f"ConvBlock_{i}",
                "params": 0,  # Will be calculated when model is built
                "output_shape": f"(?, ?, {self.config.num_filters[i]}, ?)",
                "trainable": True,
            })
        
        info.append({
            "type": "Flatten",
            "params": 0,
            "output_shape": "(?, flattened_size)",
            "trainable": False,
        })
        
        # FC layers
        hidden_dims = self.config.get("fc_layers", [256, 128])
        for i, dim in enumerate(hidden_dims):
            info.append({
                "type": f"FC_{i}",
                "params": 0,
                "output_shape": f"(?, {dim})",
                "trainable": True,
            })
        
        info.append({
            "type": "Output",
            "params": 0,
            "output_shape": f"(?, {self.config.num_classes})",
            "trainable": True,
        })
        
        return info

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        torch.save({
            "config": self.config.to_dict(),
            "state_dict": self.model.state_dict() if hasattr(self, 'model') else {},
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CNN":
        """Load a model from disk."""
        checkpoint = torch.load(filepath)
        config = CNNConfig(**checkpoint["config"])
        model = cls(config)
        
        if hasattr(model, 'model') and checkpoint["state_dict"]:
            model.model.load_state_dict(checkpoint["state_dict"])
        
        return model


# PyTorch implementation
import torch
import torch.nn as nn

class CNNPyTorch(nn.Module):
    """Pure PyTorch CNN implementation."""

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__()
        self.config = config or CNNConfig()
        
        num_blocks = len(self.config.num_filters)
        layers = []
        
        for i in range(num_blocks):
            in_channels = 3 if i == 0 else self.config.num_filters[i - 1]
            out_channels = self.config.num_filters[i]
            kernel_size = self.config.kernel_sizes[i % len(self.config.kernel_sizes)]
            pool_size = self.config.pool_sizes[i % len(self.config.pool_sizes)]
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"))
            
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(pool_size=pool_size))
            
            dropout_rate = self.config.dropout_rates[i % len(self.config.dropout_rates)]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Flatten and FC layers
        layers.append(nn.Flatten())
        
        # Calculate flattened size (will be set after first forward pass)
        self.fc_layers = nn.ModuleList()
        hidden_dims = self.config.get("fc_layers", [256, 128])
        
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                self.fc_in_dim = None  # Will be set dynamically
            else:
                prev_dim = hidden_dims[i - 1]
            
            self.fc_layers.append(nn.Linear(self.fc_in_dim or 512, dim) if self.fc_in_dim else nn.Linear(512, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], self.config.num_classes))
        
        self.features = nn.Sequential(*layers[:-len(self.fc_layers)]) if len(layers) > len(self.fc_layers) else nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Handle input shape
        if x.dim() == 4 and x.shape[1] <= 4:  # NHWC format
            x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        
        for fc in self.fc_layers:
            x = fc(x)
        
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TensorFlow/Keras implementation
class CNNKeras(keras.Model):
    """TensorFlow/Keras CNN implementation."""

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__()
        self.config = config or CNNConfig()
        
        num_blocks = len(self.config.num_filters)
        
        # Convolutional blocks
        for i in range(num_blocks):
            in_channels = 3 if i == 0 else self.config.num_filters[i - 1]
            out_channels = self.config.num_filters[i]
            kernel_size = self.config.kernel_sizes[i % len(self.config.kernel_sizes)]
            pool_size = self.config.pool_sizes[i % len(self.config.pool_sizes)]
            
            setattr(
                self, 
                f"conv{i}", 
                layers.Conv2D(out_channels, kernel_size, padding="same", input_shape=(None, None, in_channels) if i == 0 else None)
            )
            
            if self.config.use_batch_norm:
                setattr(self, f"bn{i}", layers.BatchNormalization())
            
            setattr(self, f"pool{i}", layers.MaxPooling2D(pool_size=pool_size))
            
            dropout_rate = self.config.dropout_rates[i % len(self.config.dropout_rates)]
            if dropout_rate > 0:
                setattr(self, f"dropout{i}", layers.Dropout(dropout_rate))
        
        # FC layers
        hidden_dims = self.config.get("fc_layers", [256, 128])
        
        for i, dim in enumerate(hidden_dims):
            setattr(self, f"fc{i}", layers.Dense(dim, activation="relu"))
            setattr(self, f"dropout_fc{i}", layers.Dropout(self.config.dropout_rate))
        
        self.output_layer = layers.Dense(self.config.num_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = inputs
        
        num_blocks = len(self.config.num_filters)
        for i in range(num_blocks):
            x = getattr(self, f"conv{i}")(x)
            
            if self.config.use_batch_norm:
                x = getattr(self, f"bn{i}")(x, training=training)
            
            x = tf.nn.relu(x)
            x = getattr(self, f"pool{i}")(x)
            
            dropout_rate = self.config.dropout_rates[i % len(self.config.dropout_rates)]
            if dropout_rate > 0:
                x = getattr(self, f"dropout{i}")(x, training=training)
        
        x = layers.Flatten()(x)
        
        hidden_dims = self.config.get("fc_layers", [256, 128])
        for i, dim in enumerate(hidden_dims):
            x = getattr(self, f"fc{i}")(x)
            x = getattr(self, f"dropout_fc{i}")(x, training=training)
        
        return self.output_layer(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(v.shape.num_elements() for v in self.trainable_variables)


# Register CNN models
ModelFactory.register("cnn_pytorch")(CNNPyTorch)
ModelFactory.register("cnn_keras")(CNNKeras)
