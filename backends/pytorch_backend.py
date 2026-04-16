"""
PyTorch backend implementation for ML Framework.

This module provides full PyTorch integration including:
- Model creation and management
- Optimizers and loss functions
- GPU acceleration via CUDA
- Mixed precision training with AMP
- Data loading utilities
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Iterator
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, IterableDataset
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PyTorchBackend(nn.Module):
    """PyTorch backend adapter for ML Framework.
    
    This class provides a unified interface for PyTorch operations while maintaining
    compatibility with the BackendAdapter abstract base class.
    
    Args:
        device: Device to use ('cpu', 'cuda', or 'auto'). Default is 'auto'.
        mixed_precision: Whether to enable mixed precision training (FP16).
    """

    def __init__(self, device: str = "auto", mixed_precision: bool = False):
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PyTorchBackend. Install with 'pip install torch'")
        
        self.device = self._resolve_device(device) if device == "auto" else device
        self.mixed_precision = mixed_precision
        
        # Mixed precision components
        self.scaler = GradScaler() if mixed_precision and self._has_gpu() else None
        self.amp_enabled = False

    def _resolve_device(self, requested: str) -> str:
        """Resolve the actual device to use based on availability."""
        if torch.cuda.is_available():
            return "cuda"
        print("No GPU available, using CPU")
        return "cpu"

    def _has_gpu(self) -> bool:
        """Check if a GPU is available."""
        return torch.cuda.is_available()

    @property
    def dtype(self):
        """Get the current data type (float32 or float16 for mixed precision)."""
        return torch.float16 if self.amp_enabled else torch.float32

    def create_model(
        self, 
        model_class: Any, 
        *args, 
        **kwargs
    ) -> nn.Module:
        """Create a PyTorch model instance.
        
        Args:
            model_class: The model class to instantiate (should inherit from nn.Module).
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.
        
        Returns:
            An instance of the model on the appropriate device.
        """
        if not issubclass(model_class, nn.Module):
            raise TypeError("model_class must be a subclass of torch.nn.Module")
        
        model = model_class(*args, **kwargs)
        return model.to(self.device)

    def create_optimizer(
        self, 
        model: nn.Module, 
        lr: float = 0.001, 
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        **kwargs
    ) -> optim.Optimizer:
        """Create a PyTorch optimizer for the model.
        
        Args:
            model: The model to optimize.
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', 'adagrad').
            **kwargs: Additional optimizer arguments.
        
        Returns:
            An optimizer instance.
        """
        optimizers = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
            "adamw": optim.AdamW,
        }
        
        if optimizer_type not in optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}")
        
        optimizer_class = optimizers[optimizer_type]
        
        # Prepare optimizer kwargs
        opt_kwargs = {"lr": lr, "weight_decay": weight_decay}
        opt_kwargs.update(kwargs)
        
        return optimizer_class(model.parameters(), **opt_kwargs)

    def create_loss_fn(self, loss_type: str = "cross_entropy", **kwargs) -> nn.Module:
        """Create a PyTorch loss function.
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'mse', 'mae', 'bce', 'huber').
            **kwargs: Additional loss arguments.
        
        Returns:
            A loss function instance.
        """
        losses = {
            "cross_entropy": nn.CrossEntropyLoss,
            "sparse_cross_entropy": nn.CrossEntropyLoss,  # Alias
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "bce": nn.BCELoss,
            "bce_with_logits": nn.BCEWithLogitsLoss,
            "huber": nn.HuberLoss,
            "kl_div": nn.KLDivLoss,
        }
        
        if loss_type not in losses:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")
        
        return losses[loss_type](**kwargs)

    def create_dataloader(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        batch_size: int = 32, 
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Create a PyTorch DataLoader.
        
        Args:
            X: Input features as NumPy array or tensor.
            y: Labels as NumPy array or tensor (optional).
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
            **kwargs: Additional DataLoader arguments.
        
        Returns:
            An iterator yielding batches of (X_batch, y_batch) tuples.
        """
        # Convert to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected numpy array or tensor for X, got {type(X)}")
        
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            elif not isinstance(y, torch.Tensor):
                raise TypeError(f"Expected numpy array or tensor for y, got {type(y)}")
            
            dataset = TensorDataset(X, y)
        else:
            # If no labels provided, create a simple iterable that yields X in batches
            class SimpleIterable(IterableDataset):
                def __init__(self, data, batch_size):
                    self.data = data
                    self.batch_size = batch_size
                
                def __iter__(self):
                    for i in range(0, len(self.data), self.batch_size):
                        yield self.data[i:i + self.batch_size]
            
            dataset = SimpleIterable(X, batch_size)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def to_device(self, tensor: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """Move a tensor to the specified device.
        
        Args:
            tensor: The tensor to move.
            device: Target device (defaults to self.device).
        
        Returns:
            The tensor on the target device.
        """
        if device is None:
            device = self.device
        
        return tensor.to(device)

    def compute_gradients(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for a forward-backward pass.
        
        Args:
            model: The model to compute gradients for.
            X: Input data.
            y: Target labels.
            loss_fn: Loss function (uses self.loss_fn if None).
        
        Returns:
            Dictionary of parameter names to gradient tensors.
        """
        # Forward pass
        predictions = model(X)
        
        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(predictions, y)
        else:
            raise ValueError("loss_fn must be provided")
        
        # Backward pass
        self.zero_gradients(model)
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients

    def apply_gradients(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Apply computed gradients to model parameters via optimizer.
        
        Args:
            model: The model to update.
            optimizer: The optimizer that has accumulated gradients.
        """
        optimizer.step()

    def zero_gradients(self, model: nn.Module) -> None:
        """Zero out the gradients of all parameters.
        
        Args:
            model: The model whose gradients should be zeroed.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def update_lr(self, optimizer: optim.Optimizer, lr: float) -> None:
        """Update the learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to update.
            lr: New learning rate.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get the current learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to query.
        
        Returns:
            Current learning rate (first param group's LR).
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def save_model(
        self, 
        model: nn.Module, 
        filepath: str, 
        include_optimizer: bool = False,
        optimizer: Optional[optim.Optimizer] = None,
        **kwargs
    ) -> None:
        """Save a PyTorch model to disk.
        
        Args:
            model: The model to save.
            filepath: Path where the model will be saved.
            include_optimizer: Whether to save optimizer state.
            optimizer: Optimizer instance (required if include_optimizer=True).
            **kwargs: Additional save arguments.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        save_dict = {
            "model_state_dict": model.state_dict(),
            "device": self.device,
        }
        
        if include_optimizer and optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)

    def load_model(
        self, 
        filepath: str, 
        model_class: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load a PyTorch model from disk.
        
        Args:
            filepath: Path to the saved model.
            model_class: The model class for reconstruction (required if not already known).
            device: Device to load the model onto.
            **kwargs: Additional load arguments.
        
        Returns:
            The loaded model instance.
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Create model if class provided
        if model_class is not None:
            model = model_class()
        else:
            raise ValueError("model_class must be provided for loading")
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move to device
        target_device = device or self.device
        model = model.to(target_device)
        
        return model

    def get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get the state dictionary of a model.
        
        Args:
            model: The model to extract state from.
        
        Returns:
            Dictionary containing model parameters and buffers.
        """
        return {k: v.cpu() for k, v in model.state_dict().items()}

    def set_model_state(self, model: nn.Module, state_dict: Dict[str, Any]) -> None:
        """Set the state of a model from a dictionary.
        
        Args:
            model: The model to update.
            state_dict: Dictionary containing model parameters and buffers.
        """
        # Convert numpy arrays back to tensors if needed
        converted_state = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                converted_state[k] = torch.from_numpy(v)
            else:
                converted_state[k] = v
        
        model.load_state_dict(converted_state)

    def convert_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to NumPy array.
        
        Args:
            tensor: The tensor to convert.
        
        Returns:
            NumPy array representation of the tensor.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return np.array(tensor)

    def convert_to_tensor(self, array: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
        """Convert a NumPy array to PyTorch tensor.
        
        Args:
            array: The NumPy array to convert.
            device: Device for the resulting tensor.
        
        Returns:
            PyTorch tensor.
        """
        if not isinstance(array, np.ndarray):
            return torch.tensor(array)
        
        tensor = torch.from_numpy(array)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        self.amp_enabled = True
        if self._has_gpu():
            self.scaler = GradScaler()

    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        self.amp_enabled = False
        self.scaler = None

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for mixed precision training.
        
        Args:
            loss: The loss value to scale.
        
        Returns:
            Scaled loss value (or original if not using AMP).
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_gradients(self, optimizer: optim.Optimizer) -> None:
        """Unscale gradients after backward pass in mixed precision.
        
        Args:
            optimizer: The optimizer that will update parameters.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (delegates to wrapped model)."""
        return super().forward(x)


class PyTorchModelWrapper(nn.Module):
    """Wrapper for models that need additional functionality.
    
    This wrapper can be used to add features like dropout, batch normalization,
    or custom forward logic around a base model.
    """

    def __init__(self, base_model: nn.Module, use_dropout: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dropout."""
        out = self.base_model(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LearningRateSchedulerWrapper:
    """Wrapper for learning rate scheduling.
    
    Provides a simple interface for adjusting the learning rate during training.
    """

    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

    def step(self, lr: float) -> None:
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class ExponentialLR(LearningRateSchedulerWrapper):
    """Exponential decay learning rate scheduler."""

    def __init__(self, optimizer: optim.Optimizer, initial_lr: float = 0.001, decay_rate: float = 0.95):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def step(self, epoch: int) -> None:
        """Update learning rate based on epoch."""
        lr = self.initial_lr * (self.decay_rate ** epoch)
        super().step(lr)


class CosineAnnealingLR(LearningRateSchedulerWrapper):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        initial_lr: float = 0.001, 
        T_max: int = 50,
        eta_min: float = 0.0
    ):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch: int) -> None:
        """Update learning rate based on epoch."""
        import math
        lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + 
            math.cos(math.pi * epoch / self.T_max))
        super().step(lr)
