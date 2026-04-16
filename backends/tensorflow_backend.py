"""
TensorFlow backend implementation for ML Framework.

This module provides full TensorFlow 2.x integration including:
- Model creation and management with Keras API
- Optimizers and loss functions
- GPU acceleration via CUDA/XLA
- Data loading with tf.data
- Mixed precision training support
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Iterator
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, losses
    HAS_TF = True
except ImportError:
    HAS_TF = False


class TensorFlowBackend(keras.Model):
    """TensorFlow backend adapter for ML Framework.
    
    This class provides a unified interface for TensorFlow/Keras operations while maintaining
    compatibility with the BackendAdapter abstract base class.
    
    Args:
        device: Device to use ('cpu', 'gpu', or 'auto'). Default is 'auto'.
        mixed_precision: Whether to enable mixed precision training.
    """

    def __init__(self, device: str = "auto", mixed_precision: bool = False):
        super().__init__()
        
        if not HAS_TF:
            raise ImportError("TensorFlow is required for TensorFlowBackend. Install with 'pip install tensorflow'")
        
        # Set device policy
        self._set_device(device)
        
        # Mixed precision setup
        self.mixed_precision = mixed_precision
        if mixed_precision:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision training enabled")
            except ImportError:
                print("Mixed precision not available in this TensorFlow version, using float32")

    def _set_device(self, device: str) -> None:
        """Configure the device based on user preference."""
        if device == "auto":
            # Auto-detect GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Using GPU: {gpus[0].name}")
                self.device = "gpu"
            else:
                print("No GPU available, using CPU")
                self.device = "cpu"
        elif device.lower() == "gpu":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Using GPU: {gpus[0].name}")
                self.device = "gpu"
            else:
                print("Requested GPU but none available, using CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"

    def _has_gpu(self) -> bool:
        """Check if a GPU is available."""
        return len(tf.config.list_physical_devices('GPU')) > 0

    @property
    def dtype(self):
        """Get the current data type (float16 for mixed precision, float32 otherwise)."""
        return tf.float16 if self.mixed_precision else tf.float32

    def create_model(
        self, 
        model_class: Any, 
        *args, 
        **kwargs
    ) -> keras.Model:
        """Create a TensorFlow/Keras model instance.
        
        Args:
            model_class: The model class to instantiate (should inherit from keras.Model or be a function).
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.
        
        Returns:
            An instance of the model.
        """
        if callable(model_class):
            # If it's a class, instantiate it
            if isinstance(model_class, type) and issubclass(model_class, keras.Model):
                return model_class(*args, **kwargs)
            else:
                # Assume it's a function that creates a model
                return model_class(*args, **kwargs)
        else:
            raise TypeError("model_class must be callable")

    def create_optimizer(
        self, 
        lr: float = 0.001, 
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        **kwargs
    ) -> optimizers.Optimizer:
        """Create a TensorFlow/Keras optimizer.
        
        Args:
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', 'adagrad').
            **kwargs: Additional optimizer arguments.
        
        Returns:
            An optimizer instance.
        """
        optimizers_map = {
            "adam": optimizers.Adam,
            "sgd": optimizers.SGD,
            "rmsprop": optimizers.RMSprop,
            "adagrad": optimizers.Adagrad,
            "adamw": None,  # AdamW not directly available in Keras
        }
        
        if optimizer_type not in optimizers_map:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers_map.keys())}")
        
        opt_class = optimizers_map[optimizer_type]
        
        # Prepare optimizer kwargs
        opt_kwargs = {"learning_rate": lr, "weight_decay": weight_decay}
        opt_kwargs.update(kwargs)
        
        return opt_class(**opt_kwargs)

    def create_loss_fn(self, loss_type: str = "cross_entropy", **kwargs) -> losses.Loss:
        """Create a TensorFlow/Keras loss function.
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'mse', 'mae', 'bce').
            **kwargs: Additional loss arguments.
        
        Returns:
            A loss function instance.
        """
        losses_map = {
            "cross_entropy": losses.SparseCategoricalCrossentropy,
            "sparse_cross_entropy": losses.SparseCategoricalCrossentropy,
            "categorical_cross_entropy": losses.CategoricalCrossentropy,
            "mse": losses.MeanSquaredError,
            "mae": losses.MeanAbsoluteError,
            "bce": losses.BinaryCrossentropy,
            "huber": losses.Huber,
        }
        
        if loss_type not in losses_map:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses_map.keys())}")
        
        return losses_map[loss_type](**kwargs)

    def create_dataloader(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        batch_size: int = 32, 
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ) -> Iterator[Tuple[Any, Any]]:
        """Create a TensorFlow data pipeline.
        
        Args:
            X: Input features as NumPy array or tensor.
            y: Labels as NumPy array or tensor (optional).
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of parallel workers for data loading.
            **kwargs: Additional tf.data arguments.
        
        Returns:
            An iterator yielding batches of (X_batch, y_batch) tuples.
        """
        # Create dataset from NumPy arrays
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)
        
        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def to_device(self, tensor: Any, device: Optional[str] = None) -> Any:
        """Move a tensor to the specified device.
        
        Args:
            tensor: The tensor to move.
            device: Target device (defaults to self.device).
        
        Returns:
            The tensor on the target device.
        """
        if device is None:
            device = self.device
        
        with tf.device(device):
            return tensor

    def compute_gradients(
        self, 
        model: keras.Model, 
        X: Any, 
        y: Any,
        loss_fn: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Compute gradients for a forward-backward pass.
        
        Args:
            model: The model to compute gradients for.
            X: Input data.
            y: Target labels.
            loss_fn: Loss function (uses model's loss if None).
        
        Returns:
            Dictionary of parameter names to gradient tensors.
        """
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            
            if loss_fn is not None:
                loss = loss_fn(y, predictions)
            else:
                # Use model's built-in loss computation
                loss = model.compute_loss(x=X, y=y, sample_weight=None, training=True)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        return {var.name: grad for var, grad in zip(model.trainable_variables, gradients)}

    def apply_gradients(self, model: keras.Model, optimizer: optimizers.Optimizer, gradients: Dict[str, Any]) -> None:
        """Apply computed gradients to model parameters via optimizer.
        
        Args:
            model: The model to update.
            optimizer: The optimizer instance.
            gradients: Dictionary of parameter names to gradient tensors.
        """
        # Convert dictionary back to list in correct order
        grad_list = []
        for var in model.trainable_variables:
            if var.name in gradients:
                grad_list.append(gradients[var.name])
            else:
                grad_list.append(None)  # Skip variables without gradients
        
        optimizer.apply_gradients(zip(grad_list, model.trainable_variables))

    def zero_gradients(self, model: keras.Model) -> None:
        """Zero out the gradients of all parameters.
        
        Args:
            model: The model whose gradients should be zeroed.
        """
        with tf.GradientTape() as tape:
            pass  # Empty tape to clear previous gradients
        
    def update_lr(self, optimizer: optimizers.Optimizer, lr: float) -> None:
        """Update the learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to update.
            lr: New learning rate.
        """
        optimizer.learning_rate = lr

    def get_lr(self, optimizer: optimizers.Optimizer) -> float:
        """Get the current learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to query.
        
        Returns:
            Current learning rate.
        """
        return float(optimizer.learning_rate)

    def save_model(
        self, 
        model: keras.Model, 
        filepath: str, 
        include_optimizer: bool = True,
        **kwargs
    ) -> None:
        """Save a TensorFlow/Keras model to disk.
        
        Args:
            model: The model to save.
            filepath: Path where the model will be saved.
            include_optimizer: Whether to save optimizer state (for SavedModel format).
            **kwargs: Additional save arguments.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        if filepath.endswith('.h5') or filepath.endswith('.keras'):
            model.save(filepath, save_format='h5' if filepath.endswith('.h5') else 'keras')
        elif filepath.endswith('.pb') or filepath.endswith('/'):
            # SavedModel format
            model.save(filepath, save_format='saved_model')
        else:
            # Default to HDF5
            model.save(filepath)

    def load_model(
        self, 
        filepath: str, 
        model_class: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> keras.Model:
        """Load a TensorFlow/Keras model from disk.
        
        Args:
            filepath: Path to the saved model.
            model_class: The model class for reconstruction (not needed for SavedModel).
            device: Device to load the model onto.
            **kwargs: Additional load arguments.
        
        Returns:
            The loaded model instance.
        """
        if filepath.endswith('.h5') or filepath.endswith('.keras'):
            return keras.models.load_model(filepath, compile=False)
        else:
            # SavedModel format
            return keras.models.load_model(filepath, save_format='saved_model')

    def get_model_state(self, model: keras.Model) -> Dict[str, Any]:
        """Get the state dictionary of a model.
        
        Args:
            model: The model to extract state from.
        
        Returns:
            Dictionary containing model weights.
        """
        return {var.name: var.numpy() for var in model.trainable_variables}

    def set_model_state(self, model: keras.Model, state_dict: Dict[str, Any]) -> None:
        """Set the state of a model from a dictionary.
        
        Args:
            model: The model to update.
            state_dict: Dictionary containing model weights.
        """
        for var in model.trainable_variables:
            if var.name in state_dict:
                var.assign(state_dict[var.name])

    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a TensorFlow tensor to NumPy array.
        
        Args:
            tensor: The tensor to convert.
        
        Returns:
            NumPy array representation of the tensor.
        """
        if isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        return np.array(tensor)

    def convert_to_tensor(self, array: np.ndarray, device: Optional[str] = None) -> Any:
        """Convert a NumPy array to TensorFlow tensor.
        
        Args:
            array: The NumPy array to convert.
            device: Device for the resulting tensor.
        
        Returns:
            TensorFlow tensor.
        """
        if isinstance(array, tf.Tensor):
            return array
        
        with tf.device(device or self.device):
            return tf.convert_to_tensor(array)

    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            self.mixed_precision = True
            print("Mixed precision training enabled")
        except ImportError:
            print("Mixed precision not available in this TensorFlow version")

    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('float32')
            self.mixed_precision = False
            print("Mixed precision training disabled")
        except ImportError:
            pass

    def scale_loss(self, loss: Any) -> Any:
        """Scale the loss for mixed precision training.
        
        Args:
            loss: The loss value to scale.
        
        Returns:
            Scaled loss value (or original if not using AMP).
        """
        # TensorFlow handles loss scaling automatically with mixed_float16 policy
        return loss

    def unscale_gradients(self, optimizer: Any) -> None:
        """Unscale gradients after backward pass in mixed precision.
        
        Args:
            optimizer: The optimizer that will update parameters.
        """
        # TensorFlow handles gradient scaling automatically with mixed_float16 policy
        pass


class TensorFlowModelWrapper(keras.Model):
    """Wrapper for models that need additional functionality."""

    def __init__(self, base_model: keras.Model, use_dropout: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = layers.Dropout(dropout_rate) if use_dropout else None

    def call(self, inputs, training=False):
        """Forward pass with optional dropout."""
        out = self.base_model(inputs, training=training)
        if self.dropout is not None:
            out = self.dropout(out, training=training)
        return out


class LearningRateScheduleWrapper:
    """Wrapper for learning rate scheduling in TensorFlow."""

    def __init__(self, initial_lr: float = 0.001):
        self.initial_lr = initial_lr

    def get_schedule(self, num_epochs: int) -> keras.optimizers.schedules.LearningRateSchedule:
        """Get a learning rate schedule for the given number of epochs."""
        raise NotImplementedError


class ExponentialDecaySchedule(LearningRateScheduleWrapper):
    """Exponential decay learning rate scheduler."""

    def __init__(self, initial_lr: float = 0.001, decay_rate: float = 0.95, staircase: bool = False):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.staircase = staircase

    def get_schedule(self, num_epochs: int) -> keras.optimizers.schedules.LearningRateSchedule:
        """Get exponential decay schedule."""
        return optimizers.exponential_decay(
            self.initial_lr,
            global_step=tf.Variable(0, trainable=False),
            decay_steps=num_epochs,
            decay_rate=self.decay_rate,
            staircase=self.staircase
        )


class CosineDecaySchedule(LearningRateScheduleWrapper):
    """Cosine decay learning rate scheduler."""

    def __init__(self, initial_lr: float = 0.001, T_0: int = 10, T_mult: int = 2, eta_min: float = 0.0):
        super().__init__(initial_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def get_schedule(self, num_epochs: int) -> keras.optimizers.schedules.LearningRateSchedule:
        """Get cosine decay schedule with restarts."""
        return optimizers.schedules.CosineDecayRestarts(
            self.initial_lr,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min
        )


class WarmUpSchedule(LearningRateScheduleWrapper):
    """Warmup learning rate scheduler."""

    def __init__(self, initial_lr: float = 0.001, warmup_steps: int = 1000, peak_lr: float = 0.01):
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr

    def get_schedule(self, num_epochs: int) -> keras.optimizers.schedules.LearningRateSchedule:
        """Get linear warmup schedule."""
        return optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[self.warmup_steps],
            values=[self.peak_lr, self.initial_lr]
        )


class ModelCheckpointCallback(keras.callbacks.Callback):
    """Keras callback for model checkpointing."""

    def __init__(
        self, 
        filepath: str = "checkpoint_{epoch}.keras",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = None

    def on_epoch_end(self, epoch: int, logs=None):
        """Save model at end of each epoch if monitored metric improves."""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.save_best_only:
            if self.best is None or (self.monitor.startswith("acc") and current > self.best):
                self.best = current
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            elif current < self.best:
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                self.model.save(filepath)
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve")
        else:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.model.save(filepath)


class EarlyStoppingCallback(keras.callbacks.Callback):
    """Keras callback for early stopping."""

    def __init__(
        self, 
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch: int, logs=None):
        """Check if training should stop based on monitored metric."""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # Check for improvement
        if self.best_weights is None or (self.monitor.startswith("acc") and current > self.best + self.min_delta):
            self.best = current
            self.epochs_without_improvement = 0
            self.best_weights = self.model.get_weights()
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            print(f"\nEarlyStopping triggered at epoch {epoch + 1}")
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print("Restored best model weights")
            raise KeyboardInterrupt()


class TensorBoardCallback(keras.callbacks.Callback):
    """Keras callback for TensorBoard logging."""

    def __init__(self, log_dir: str = "logs", write_graph: bool = True):
        super().__init__()
        self.log_dir = log_dir
        self.write_graph = write_graph


class LearningRateSchedulerCallback(keras.callbacks.Callback):
    """Keras callback for learning rate scheduling."""

    def __init__(self, lr_schedule):
        super().__init__()
        self.lr_schedule = lr_schedule

    def on_epoch_begin(self, epoch, logs=None):
        """Update learning rate at the beginning of each epoch."""
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        
        if isinstance(self.lr_schedule, keras.optimizers.schedules.LearningRateSchedule):
            new_lr = self.lr_schedule(epoch)
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            
            if hasattr(self, 'verbose') and self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Learning rate scheduler set learning rate to {new_lr:.6f}")
