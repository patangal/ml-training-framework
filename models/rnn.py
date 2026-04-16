"""
Recurrent Neural Network (RNN) implementations for ML Framework.

This module provides RNN, LSTM, and GRU architectures suitable for:
- Sequence modeling tasks
- Time series prediction
- Natural language processing
- Speech recognition
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class RNNConfig(ModelConfig):
    """Configuration class for RNN models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default values specific to RNNs
        defaults = {
            "input_dim": 128,
            "hidden_dims": [64],
            "num_classes": 10,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "use_bidirectional": False,
            "return_sequences": True,
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
        return self.config.get("hidden_dims", [64])

    @property
    def num_classes(self) -> int:
        """Get number of output classes (for classification)."""
        return self.config.get("num_classes", 10)

    @property
    def dropout(self) -> float:
        """Get dropout rate for dense layers."""
        return self.config.get("dropout", 0.2)

    @property
    def recurrent_dropout(self) -> float:
        """Get dropout rate for recurrent connections."""
        return self.config.get("recurrent_dropout", 0.2)

    @property
    def use_bidirectional(self) -> bool:
        """Check if bidirectional RNN is used."""
        return self.config.get("use_bidirectional", False)

    @property
    def return_sequences(self) -> bool:
        """Check if all layers return sequences."""
        return self.config.get("return_sequences", True)


class RNN(BaseModel):
    """Base Recurrent Neural Network class.
    
    This is a base class that provides common functionality for RNN variants.
    Subclasses (LSTM, GRU) inherit from this class and provide specific implementations.
    """

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__(config)
        
        self.config = config or RNNConfig()
        self._layers: List[Any] = []

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the RNN architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features).
        """
        if len(input_shape) == 2:
            # Assume single feature per timestep
            input_shape = (*input_shape, 1)
        
        self.input_shape = input_shape
        
        num_layers = len(self.config.hidden_dims)
        
        for i in range(num_layers):
            layer_name = f"rnn_layer_{i}"
            
            if i == 0:
                input_dim = self.config.input_dim
            else:
                input_dim = self.config.hidden_dims[i - 1]
            
            hidden_dim = self.config.hidden_dims[i]
            
            # Create RNN layer (to be implemented by subclasses)
            rnn_layer = self._create_rnn_layer(input_dim, hidden_dim, i == num_layers - 1)
            self._layers.append((layer_name, rnn_layer))

        # Output layer for classification
        output_input_dim = self.config.hidden_dims[-1] if not self.config.use_bidirectional else self.config.hidden_dims[-1] * 2
        
        if self.config.return_sequences:
            # For sequence-to-sequence tasks
            self._layers.append(("output", nn.Linear(output_input_dim, self.config.num_classes)))
        else:
            # For sequence-to-label tasks
            self._layers.append(("dense", nn.Linear(output_input_dim, self.config.hidden_dims[-1])))
            self._layers.append(("dropout_out", nn.Dropout(self.config.dropout)))
            self._layers.append(("output", nn.Linear(self.config.hidden_dims[-1], self.config.num_classes)))

    def _create_rnn_layer(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        is_last_layer: bool
    ) -> Any:
        """Create an RNN layer. To be overridden by subclasses."""
        raise NotImplementedError


class LSTM(RNN):
    """Long Short-Term Memory (LSTM) network.
    
    LSTMs are a type of RNN that can learn long-term dependencies, making them
    particularly effective for sequence modeling tasks.
    
    Example usage:
        config = LSTMConfig(
            input_dim=128,
            hidden_dims=[64, 128],
            num_classes=10,
            use_bidirectional=True,
        )
        model = LSTM(config)
    """

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__(config)
        
        self._lstm_layers: List[Any] = []

    def _create_rnn_layer(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        is_last_layer: bool
    ) -> Any:
        """Create an LSTM layer."""
        layers = []
        
        # Bidirectional wrapper if enabled
        if self.config.use_bidirectional and len(self._lstm_layers) == 0:
            layers.append(("bidir", nn.LSTM(input_dim, hidden_dim, batch_first=True)))
        else:
            layers.append(("lstm", nn.LSTM(input_dim, hidden_dim, batch_first=True)))
        
        # Dropout for recurrent connections
        if self.config.recurrent_dropout > 0 and is_last_layer:
            layers.append(("recurrent_dropout", nn.Dropout(self.config.recurrent_dropout)))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features).
        
        Returns:
            Output tensor of shape (batch_size, num_classes) or 
            (batch_size, sequence_length, num_classes) for sequence outputs.
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        output, (h_n, c_n) = self._lstm_layers[0](x)
        
        for i in range(1, len(self._lstm_layers)):
            output, _ = self._lstm_layers[i](output)
        
        if not self.config.return_sequences:
            # Use the last hidden state
            output = output[:, -1, :]
        
        # Output layers
        for _, layer in self._layers[-3:]:  # Skip LSTM layers
            output = layer(output)
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if not hasattr(self, 'model'):
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer."""
        info = []
        
        num_layers = len(self.config.hidden_dims)
        for i in range(num_layers):
            hidden_dim = self.config.hidden_dims[i]
            
            if self.config.use_bidirectional and i == 0:
                info.append({
                    "type": f"BiLSTM_{i}",
                    "params": 4 * hidden_dim * (self.config.input_dim + hidden_dim) + 4 * hidden_dim,
                    "output_shape": f"(?, ?, {hidden_dim * 2})" if self.config.return_sequences else f"(?, {hidden_dim * 2})",
                    "trainable": True,
                })
            else:
                info.append({
                    "type": f"LSTM_{i}",
                    "params": 4 * hidden_dim * (self.config.input_dim if i == 0 else self.config.hidden_dims[i - 1] + hidden_dim) + 4 * hidden_dim,
                    "output_shape": f"(?, ?, {hidden_dim})" if self.config.return_sequences else f"(?, {hidden_dim})",
                    "trainable": True,
                })
        
        if not self.config.return_sequences:
            info.append({
                "type": "Dense",
                "params": self.config.hidden_dims[-1] * self.config.num_classes,
                "output_shape": f"(?, {self.config.num_classes})",
                "trainable": True,
            })
        
        return info


class GRU(RNN):
    """Gated Recurrent Unit (GRU) network.
    
    GRUs are similar to LSTMs but have a simpler architecture with fewer parameters.
    They often train faster while maintaining comparable performance.
    
    Example usage:
        config = RNNConfig(
            input_dim=128,
            hidden_dims=[64],
            num_classes=10,
            use_bidirectional=True,
        )
        model = GRU(config)
    """

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__(config)
        
        self._gru_layers: List[Any] = []

    def _create_rnn_layer(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        is_last_layer: bool
    ) -> Any:
        """Create a GRU layer."""
        layers = []
        
        # Bidirectional wrapper if enabled
        if self.config.use_bidirectional and len(self._gru_layers) == 0:
            layers.append(("bidir", nn.GRU(input_dim, hidden_dim, batch_first=True)))
        else:
            layers.append(("gru", nn.GRU(input_dim, hidden_dim, batch_first=True)))
        
        # Dropout for recurrent connections
        if self.config.recurrent_dropout > 0 and is_last_layer:
            layers.append(("recurrent_dropout", nn.Dropout(self.config.recurrent_dropout)))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features).
        
        Returns:
            Output tensor of shape (batch_size, num_classes) or 
            (batch_size, sequence_length, num_classes) for sequence outputs.
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        output, h_n = self._gru_layers[0](x)
        
        for i in range(1, len(self._gru_layers)):
            output, _ = self._gru_layers[i](output)
        
        if not self.config.return_sequences:
            # Use the last hidden state
            output = output[:, -1, :]
        
        # Output layers
        for _, layer in self._layers[-3:]:  # Skip GRU layers
            output = layer(output)
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if not hasattr(self, 'model'):
            return 0
        
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class RNNConfigExtended(RNNConfig):
    """Extended configuration class with additional options."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        defaults = {
            "layer_norm": False,
            "peephole_connections": False,  # LSTM specific
            "forget_bias": 1.0,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def layer_norm(self) -> bool:
        """Check if layer normalization is used."""
        return self.config.get("layer_norm", False)


# PyTorch implementations

class LSTMPyTorch(nn.Module):
    """Pure PyTorch LSTM implementation."""

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        
        num_layers = len(self.config.hidden_dims)
        
        # Bidirectional flag
        self.bidirectional = self.config.use_bidirectional
        
        # LSTM layers
        input_dim = self.config.input_dim
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            layer = nn.LSTM(
                input_dim, 
                hidden_dim,
                num_layers=1 if not self.bidirectional else 2,
                batch_first=True,
                dropout=self.config.recurrent_dropout if i < num_layers - 1 else 0
            )
            setattr(self, f"lstm_{i}", layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)
        
        # Output layers
        output_dim = self.config.hidden_dims[-1] * (2 if self.bidirectional else 1)
        
        if not self.config.return_sequences:
            self.fc = nn.Linear(output_dim, self.config.hidden_dims[-1])
            self.dropout_out = nn.Dropout(self.config.dropout)
            self.output_layer = nn.Linear(self.config.hidden_dims[-1], self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        output = x
        
        for i in range(len(self.config.hidden_dims)):
            lstm_layer = getattr(self, f"lstm_{i}")
            output, _ = lstm_layer(output)
        
        if not self.config.return_sequences:
            output = output[:, -1, :]
            output = self.fc(output)
            output = self.dropout_out(output)
            output = self.output_layer(output)
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUPyTorch(nn.Module):
    """Pure PyTorch GRU implementation."""

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        
        num_layers = len(self.config.hidden_dims)
        self.bidirectional = self.config.use_bidirectional
        
        # GRU layers
        input_dim = self.config.input_dim
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            layer = nn.GRU(
                input_dim, 
                hidden_dim,
                num_layers=1 if not self.bidirectional else 2,
                batch_first=True,
                dropout=self.config.recurrent_dropout if i < num_layers - 1 else 0
            )
            setattr(self, f"gru_{i}", layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)
        
        # Output layers
        output_dim = self.config.hidden_dims[-1] * (2 if self.bidirectional else 1)
        
        if not self.config.return_sequences:
            self.fc = nn.Linear(output_dim, self.config.hidden_dims[-1])
            self.dropout_out = nn.Dropout(self.config.dropout)
            self.output_layer = nn.Linear(self.config.hidden_dims[-1], self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        output = x
        
        for i in range(len(self.config.hidden_dims)):
            gru_layer = getattr(self, f"gru_{i}")
            output, _ = gru_layer(output)
        
        if not self.config.return_sequences:
            output = output[:, -1, :]
            output = self.fc(output)
            output = self.dropout_out(output)
            output = self.output_layer(output)
        
        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TensorFlow/Keras implementations

class LSTMKeras(keras.Model):
    """TensorFlow/Keras LSTM implementation."""

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        
        num_layers = len(self.config.hidden_dims)
        
        # LSTM layers
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            if self.config.use_bidirectional and i == 0:
                setattr(
                    self, 
                    f"lstm_{i}", 
                    layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=(i < num_layers - 1)))
                )
            else:
                setattr(
                    self, 
                    f"lstm_{i}", 
                    layers.LSTM(hidden_dim, return_sequences=(i < num_layers - 1))
                )
        
        # Output layers
        if not self.config.return_sequences:
            self.fc = layers.Dense(self.config.hidden_dims[-1], activation="relu")
            self.dropout_out = layers.Dropout(self.config.dropout)
            self.output_layer = layers.Dense(self.config.num_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = inputs
        
        for i in range(len(self.config.hidden_dims)):
            lstm_layer = getattr(self, f"lstm_{i}")
            x = lstm_layer(x)
        
        if not self.config.return_sequences:
            x = self.fc(x)
            x = self.dropout_out(x, training=training)
            return self.output_layer(x)
        
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(v.shape.num_elements() for v in self.trainable_variables)


class GRUKeras(keras.Model):
    """TensorFlow/Keras GRU implementation."""

    def __init__(self, config: Optional[RNNConfig] = None):
        super().__init__()
        self.config = config or RNNConfig()
        
        num_layers = len(self.config.hidden_dims)
        
        # GRU layers
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            if self.config.use_bidirectional and i == 0:
                setattr(
                    self, 
                    f"gru_{i}", 
                    layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=(i < num_layers - 1)))
                )
            else:
                setattr(
                    self, 
                    f"gru_{i}", 
                    layers.GRU(hidden_dim, return_sequences=(i < num_layers - 1))
                )
        
        # Output layers
        if not self.config.return_sequences:
            self.fc = layers.Dense(self.config.hidden_dims[-1], activation="relu")
            self.dropout_out = layers.Dropout(self.config.dropout)
            self.output_layer = layers.Dense(self.config.num_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = inputs
        
        for i in range(len(self.config.hidden_dims)):
            gru_layer = getattr(self, f"gru_{i}")
            x = gru_layer(x)
        
        if not self.config.return_sequences:
            x = self.fc(x)
            x = self.dropout_out(x, training=training)
            return self.output_layer(x)
        
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(v.shape.num_elements() for v in self.trainable_variables)


# Register RNN models
ModelFactory.register("lstm_pytorch")(LSTMPyTorch)
ModelFactory.register("gru_pytorch")(GRUPyTorch)
ModelFactory.register("lstm_keras")(LSTMKeras)
ModelFactory.register("gru_keras")(GRUKeras)
