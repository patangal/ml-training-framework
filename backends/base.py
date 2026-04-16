"""
Base backend adapter for ML Framework.

This module provides an abstract base class that defines the interface for
backend adapters (PyTorch, TensorFlow). Concrete implementations must inherit
from this class and implement all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Iterator


class BackendAdapter(ABC):
    """Abstract base class for backend adapters.
    
    This class defines the interface that all backend implementations must follow.
    It provides a unified API for model operations regardless of the underlying
    framework (PyTorch or TensorFlow).
    """

    def __init__(self, device: str = "auto", mixed_precision: bool = False):
        """Initialize the backend adapter.
        
        Args:
            device: Device to use for computation ('cpu', 'cuda', 'gpu', or 'auto').
            mixed_precision: Whether to use mixed precision training (FP16).
        """
        self.device = self._resolve_device(device) if device == "auto" else device
        self.mixed_precision = mixed_precision
        
        # Mixed precision scaler for automatic loss scaling
        self.scaler = None

    def _resolve_device(self, requested: str) -> str:
        """Resolve the actual device to use based on availability.
        
        Args:
            requested: The requested device type.
        
        Returns:
            The resolved device string ('cpu', 'cuda', etc.).
        """
        # Check for GPU availability
        if self._has_gpu():
            return "cuda"
        else:
            print("No GPU available, using CPU")
            return "cpu"

    @abstractmethod
    def _has_gpu(self) -> bool:
        """Check if a GPU is available."""
        pass

    @abstractmethod
    def create_model(
        self, 
        model_class: Any, 
        *args, 
        **kwargs
    ) -> Any:
        """Create a model instance.
        
        Args:
            model_class: The model class to instantiate.
            *args: Positional arguments for the model constructor.
            **kwargs: Keyword arguments for the model constructor.
        
        Returns:
            An instance of the model.
        """
        pass

    @abstractmethod
    def create_optimizer(
        self, 
        model: Any, 
        lr: float = 0.001, 
        weight_decay: float = 0.0,
        **kwargs
    ) -> Any:
        """Create an optimizer for the model.
        
        Args:
            model: The model to optimize.
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            **kwargs: Additional optimizer arguments.
        
        Returns:
            An optimizer instance.
        """
        pass

    @abstractmethod
    def create_loss_fn(self, loss_type: str = "cross_entropy", **kwargs) -> Any:
        """Create a loss function.
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'mse', 'mae', etc.).
            **kwargs: Additional loss arguments.
        
        Returns:
            A loss function instance.
        """
        pass

    @abstractmethod
    def create_dataloader(
        self, 
        X: Any, 
        y: Optional[Any] = None, 
        batch_size: int = 32, 
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ) -> Iterator[Tuple[Any, Any]]:
        """Create a data loader for training or evaluation.
        
        Args:
            X: Input features.
            y: Labels (optional).
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
            **kwargs: Additional DataLoader arguments.
        
        Returns:
            An iterator yielding batches of (X_batch, y_batch) tuples.
        """
        pass

    @abstractmethod
    def to_device(self, tensor: Any, device: Optional[str] = None) -> Any:
        """Move a tensor to the specified device.
        
        Args:
            tensor: The tensor to move.
            device: Target device (defaults to self.device).
        
        Returns:
            The tensor on the target device.
        """
        pass

    @abstractmethod
    def compute_gradients(self, model: Any, X: Any, y: Any) -> Dict[str, Any]:
        """Compute gradients for a forward-backward pass.
        
        Args:
            model: The model to compute gradients for.
            X: Input data.
            y: Target labels.
        
        Returns:
            Dictionary of parameter names to gradient tensors.
        """
        pass

    @abstractmethod
    def apply_gradients(self, model: Any, gradients: Dict[str, Any]) -> None:
        """Apply computed gradients to model parameters.
        
        Args:
            model: The model to update.
            gradients: Dictionary of parameter names to gradient tensors.
        """
        pass

    @abstractmethod
    def zero_gradients(self, model: Any) -> None:
        """Zero out the gradients of all parameters.
        
        Args:
            model: The model whose gradients should be zeroed.
        """
        pass

    @abstractmethod
    def update_lr(self, optimizer: Any, lr: float) -> None:
        """Update the learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to update.
            lr: New learning rate.
        """
        pass

    @abstractmethod
    def get_lr(self, optimizer: Any) -> float:
        """Get the current learning rate of an optimizer.
        
        Args:
            optimizer: The optimizer to query.
        
        Returns:
            Current learning rate.
        """
        pass

    @abstractmethod
    def save_model(
        self, 
        model: Any, 
        filepath: str, 
        include_optimizer: bool = True,
        **kwargs
    ) -> None:
        """Save a model to disk.
        
        Args:
            model: The model to save.
            filepath: Path where the model will be saved.
            include_optimizer: Whether to save optimizer state.
            **kwargs: Additional save arguments.
        """
        pass

    @abstractmethod
    def load_model(
        self, 
        filepath: str, 
        model_class: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
            model_class: The model class for reconstruction (if needed).
            device: Device to load the model onto.
            **kwargs: Additional load arguments.
        
        Returns:
            The loaded model instance.
        """
        pass

    @abstractmethod
    def get_model_state(self, model: Any) -> Dict[str, Any]:
        """Get the state dictionary of a model.
        
        Args:
            model: The model to extract state from.
        
        Returns:
            Dictionary containing model parameters and buffers.
        """
        pass

    @abstractmethod
    def set_model_state(self, model: Any, state_dict: Dict[str, Any]) -> None:
        """Set the state of a model from a dictionary.
        
        Args:
            model: The model to update.
            state_dict: Dictionary containing model parameters and buffers.
        """
        pass

    @abstractmethod
    def convert_to_numpy(self, tensor: Any) -> Any:
        """Convert a framework-specific tensor to NumPy array.
        
        Args:
            tensor: The tensor to convert.
        
        Returns:
            NumPy array representation of the tensor.
        """
        pass

    @abstractmethod
    def convert_to_tensor(self, array: Any, device: Optional[str] = None) -> Any:
        """Convert a NumPy array to framework-specific tensor.
        
        Args:
            array: The NumPy array to convert.
            device: Device for the resulting tensor.
        
        Returns:
            Framework-specific tensor.
        """
        pass

    @abstractmethod
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        pass

    @abstractmethod
    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        pass

    @abstractmethod
    def scale_loss(self, loss: Any) -> Any:
        """Scale the loss for mixed precision training.
        
        Args:
            loss: The loss value to scale.
        
        Returns:
            Scaled loss value.
        """
        pass

    @abstractmethod
    def unscale_gradients(self, optimizer: Any) -> None:
        """Unscale gradients after backward pass in mixed precision.
        
        Args:
            optimizer: The optimizer that will update parameters.
        """
        pass


class BackendRegistry:
    """Registry for backend implementations."""
    
    _backends: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a backend class.
        
        Args:
            name: The name to register the backend under.
        
        Returns:
            Decorator function.
        """
        def decorator(klass):
            cls._backends[name] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a registered backend class by name.
        
        Args:
            name: The name of the backend.
        
        Returns:
            The backend class.
        
        Raises:
            ValueError: If no backend with the given name is registered.
        """
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}. Available backends: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends.
        
        Returns:
            List of backend names.
        """
        return list(cls._backends.keys())


# Register built-in backends
from .pytorch_backend import PyTorchBackend
from .tensorflow_backend import TensorFlowBackend

BackendRegistry.register("pytorch")(PyTorchBackend)
BackendRegistry.register("tensorflow")(TensorFlowBackend)
BackendRegistry.register("tf")(TensorFlowBackend)  # Alias
