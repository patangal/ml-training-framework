"""
Checkpoint and serialization utilities for ML Framework.

This module provides functionality for:
- Saving and loading model checkpoints during training
- Serializing models to various formats (PyTorch, TensorFlow, ONNX)
- Managing checkpoint directories and cleanup
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import shutil
import numpy as np


class CheckpointManager:
    """Manages model checkpoints during training.
    
    This class handles saving and loading model checkpoints at specified intervals,
    keeping track of the best models based on monitored metrics.
    
    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric name to monitor for best model selection.
        save_best_only: Whether to only save the best model.
        save_every_n_epochs: Save checkpoint every N epochs.
        keep_last_n: Number of recent checkpoints to keep.
        verbose: Verbosity level.
    """

    def __init__(
        self, 
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        save_every_n_epochs: int = 1,
        keep_last_n: int = 3,
        verbose: int = 1,
    ):
        """Initialize the checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric name to monitor for best model selection.
            save_best_only: Whether to only save the best model.
            save_every_n_epochs: Save checkpoint every N epochs.
            keep_last_n: Number of recent checkpoints to keep.
            verbose: Verbosity level.
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Track best score and checkpoint history
        self.best_score = None
        self.checkpoint_history: List[Dict[str, Any]] = []

    def save_checkpoint(
        self, 
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[Any] = None,
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint for the current training state.
        
        Args:
            model: Model instance to save.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.
            optimizer: Optimizer instance (optional).
            additional_state: Additional state dictionary (optional).
        
        Returns:
            Path to the saved checkpoint file.
        """
        # Determine if this is a best model
        should_save_best = False
        
        if self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            if self.save_best_only:
                if self.best_score is None:
                    should_save_best = True
                    self.best_score = current_score
                elif self._is_improvement(current_score):
                    should_save_best = True
                    self.best_score = current_score
            
            # Also save periodically
            if epoch % self.save_every_n_epochs == 0 and not self.save_best_only:
                should_save_best = True
        
        if not should_save_best:
            return ""
        
        # Generate checkpoint filename
        timestamp = f"epoch_{epoch+1}"
        score_str = f"{metrics.get(self.monitor, 0):.4f}" if self.monitor in metrics else "best"
        filename = f"checkpoint_{timestamp}_{score_str}.pt"
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch + 1,
            "metrics": metrics,
            "model_state_dict": self._get_model_state(model),
        }
        
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = self._get_optimizer_state(optimizer)
        
        if additional_state is not None:
            checkpoint_data["additional_state"] = additional_state
        
        # Save checkpoint
        try:
            import torch
            torch.save(checkpoint_data, filepath)
            
            # Track history
            self.checkpoint_history.append({
                "epoch": epoch + 1,
                "filepath": filepath,
                "score": metrics.get(self.monitor, 0),
            })
            
            if self.verbose > 0:
                print(f"Saved checkpoint to {filename}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return filepath
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save checkpoint: {e}")
            return ""

    def load_checkpoint(
        self, 
        model: Any,
        optimizer: Optional[Any] = None,
        filepath: Optional[str] = None,
        map_location: str = "cpu",
    ) -> Tuple[int, Dict[str, float]]:
        """Load a checkpoint and restore training state.
        
        Args:
            model: Model instance to load weights into.
            optimizer: Optimizer instance to restore (optional).
            filepath: Path to checkpoint file (uses latest if None).
            map_location: Device to load tensors onto.
        
        Returns:
            Tuple of (start_epoch, metrics_dict).
        """
        # Find latest checkpoint if not specified
        if filepath is None:
            filepath = self._get_latest_checkpoint()
            if filepath is None:
                raise FileNotFoundError("No checkpoint found")
        
        try:
            import torch
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=map_location)
            
            # Restore model state
            model_state = checkpoint.get("model_state_dict", {})
            self._set_model_state(model, model_state)
            
            # Restore optimizer state if provided
            if optimizer is not None:
                opt_state = checkpoint.get("optimizer_state_dict", {})
                self._set_optimizer_state(optimizer, opt_state)
            
            # Get epoch and metrics
            start_epoch = checkpoint.get("epoch", 0)
            metrics = checkpoint.get("metrics", {})
            
            if self.verbose > 0:
                print(f"Loaded checkpoint from {os.path.basename(filepath)}")
                print(f"Resuming from epoch {start_epoch}")
            
            return start_epoch, metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def _is_improvement(self, current_score: float) -> bool:
        """Check if the current score is an improvement.
        
        Args:
            current_score: Current metric value.
        
        Returns:
            True if this represents an improvement.
        """
        # For loss metrics, lower is better
        if "loss" in self.monitor.lower() or "error" in self.monitor.lower():
            return current_score < (self.best_score or float('inf'))
        else:
            # For other metrics, higher is better
            return current_score > (self.best_score or float('-inf'))

    def _get_model_state(self, model: Any) -> Dict[str, Any]:
        """Extract state from a model.
        
        Args:
            model: Model instance.
        
        Returns:
            Dictionary containing model state.
        """
        if hasattr(model, 'state_dict'):
            return {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        elif hasattr(model, 'get_weights'):
            return {f"weight_{i}": w for i, w in enumerate(model.get_weights())}
        else:
            # Generic fallback
            return {"state": str(type(model))}

    def _set_model_state(self, model: Any, state_dict: Dict[str, Any]) -> None:
        """Set state on a model.
        
        Args:
            model: Model instance to update.
            state_dict: Dictionary containing model state.
        """
        if hasattr(model, 'load_state_dict'):
            import torch
            converted = {k: torch.from_numpy(v) for k, v in state_dict.items()}
            model.load_state_dict(converted)
        elif hasattr(model, 'set_weights'):
            weights = list(state_dict.values())
            model.set_weights(weights)

    def _get_optimizer_state(self, optimizer: Any) -> Dict[str, Any]:
        """Extract state from an optimizer.
        
        Args:
            optimizer: Optimizer instance.
        
        Returns:
            Dictionary containing optimizer state.
        """
        if hasattr(optimizer, 'state_dict'):
            return optimizer.state_dict()
        else:
            return {}

    def _set_optimizer_state(self, optimizer: Any, state_dict: Dict[str, Any]) -> None:
        """Set state on an optimizer.
        
        Args:
            optimizer: Optimizer instance to update.
            state_dict: Dictionary containing optimizer state.
        """
        if hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(state_dict)

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if none exist.
        """
        if not os.path.exists(self.save_dir):
            return None
        
        files = [f for f in os.listdir(self.save_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        
        if not files:
            return None
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)), reverse=True)
        
        return os.path.join(self.save_dir, files[0])

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the keep limit.
        
        Keeps only the most recent N checkpoints and the best model.
        """
        if len(self.checkpoint_history) <= self.keep_last_n:
            return
        
        # Get paths of checkpoints to remove (oldest ones)
        files_to_remove = [
            h["filepath"] for h in self.checkpoint_history[:-self.keep_last_n]
        ]
        
        for filepath in files_to_remove:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    # Also remove associated metadata file
                    meta_path = filepath.replace(".pt", ".json")
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to remove old checkpoint {filepath}: {e}")


class ModelSerializer:
    """Handles model serialization to various formats.
    
    Supports saving models in multiple formats including:
    - PyTorch (.pt, .pth)
    - TensorFlow SavedModel (.pb, /directory/)
    - ONNX (.onnx)
    - JSON configuration files
    """

    @staticmethod
    def save_pytorch(model: Any, filepath: str, include_optimizer: bool = False) -> None:
        """Save a PyTorch model.
        
        Args:
            model: PyTorch model instance.
            filepath: Path to save the model.
            include_optimizer: Whether to include optimizer state.
        """
        import torch
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": type(model).__name__,
        }
        
        if include_optimizer:
            # This would require access to the optimizer instance
            pass
        
        torch.save(save_dict, filepath)

    @staticmethod
    def load_pytorch(filepath: str, model_class: Any, map_location: str = "cpu") -> Any:
        """Load a PyTorch model.
        
        Args:
            filepath: Path to the saved model.
            model_class: Class of the model to instantiate.
            map_location: Device to load tensors onto.
        
        Returns:
            Loaded model instance.
        """
        import torch
        
        checkpoint = torch.load(filepath, map_location=map_location)
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model

    @staticmethod
    def save_tensorflow(model: Any, filepath: str) -> None:
        """Save a TensorFlow/Keras model.
        
        Args:
            model: TensorFlow/Keras model instance.
            filepath: Path to save the model.
        """
        import tensorflow as tf
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        if filepath.endswith(".pb") or not "." in os.path.basename(filepath):
            # SavedModel format
            model.save(filepath, save_format="saved_model")
        else:
            # HDF5/Keras format
            model.save(filepath)

    @staticmethod
    def load_tensorflow(filepath: str) -> Any:
        """Load a TensorFlow/Keras model.
        
        Args:
            filepath: Path to the saved model.
        
        Returns:
            Loaded model instance.
        """
        import tensorflow as tf
        
        return tf.keras.models.load_model(filepath)

    @staticmethod
    def save_onnx(model: Any, input_shape: Tuple[int, ...], filepath: str) -> None:
        """Export a model to ONNX format.
        
        Args:
            model: Model instance (PyTorch or TensorFlow).
            input_shape: Shape of input tensor for tracing.
            filepath: Path to save the ONNX file.
        """
        import torch
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    @staticmethod
    def save_config(model: Any, config: Dict[str, Any], filepath: str) -> None:
        """Save model configuration to JSON.
        
        Args:
            model: Model instance (for extracting architecture info).
            config: Configuration dictionary to save.
            filepath: Path to save the configuration file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # Add metadata
        config["metadata"] = {
            "model_class": type(model).__name__,
            "timestamp": str(np.datetime64('now')),
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def load_config(filepath: str) -> Dict[str, Any]:
        """Load model configuration from JSON.
        
        Args:
            filepath: Path to the configuration file.
        
        Returns:
            Configuration dictionary.
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def export_all(
        model: Any, 
        model_class: Any,
        input_shape: Tuple[int, ...],
        output_dir: str,
        include_optimizer: bool = False,
    ) -> List[str]:
        """Export model in all supported formats.
        
        Args:
            model: Model instance to export.
            model_class: Class of the model.
            input_shape: Shape of input tensor.
            output_dir: Directory to save exports.
            include_optimizer: Whether to include optimizer state (PyTorch only).
        
        Returns:
            List of paths to exported files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        export_paths = []
        
        # PyTorch format
        pt_path = os.path.join(output_dir, "model.pt")
        ModelSerializer.save_pytorch(model, pt_path, include_optimizer=include_optimizer)
        export_paths.append(pt_path)
        
        # ONNX format
        onnx_path = os.path.join(output_dir, "model.onnx")
        try:
            ModelSerializer.save_onnx(model, input_shape, onnx_path)
            export_paths.append(onnx_path)
        except Exception as e:
            print(f"Failed to export ONNX: {e}")
        
        # Configuration file
        config = {
            "input_shape": list(input_shape),
            "model_class": model_class.__name__,
        }
        config_path = os.path.join(output_dir, "config.json")
        ModelSerializer.save_config(model, config, config_path)
        export_paths.append(config_path)
        
        return export_paths


class CheckpointMetadata:
    """Stores metadata about checkpoints."""

    def __init__(self):
        """Initialize checkpoint metadata."""
        self.checkpoints: List[Dict[str, Any]] = []

    def add_checkpoint(
        self, 
        filepath: str, 
        epoch: int, 
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Add a checkpoint entry.
        
        Args:
            filepath: Path to the checkpoint file.
            epoch: Epoch number when saved.
            metrics: Metrics at save time.
            is_best: Whether this is the best model so far.
        """
        self.checkpoints.append({
            "filepath": filepath,
            "epoch": epoch,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": str(np.datetime64('now')),
        })

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint based on monitored metric.
        
        Returns:
            Best checkpoint entry or None.
        """
        if not self.checkpoints:
            return None
        
        # Find checkpoint with best score for monitored metric
        best = max(
            (c for c in self.checkpoints if c.get("is_best")),
            key=lambda x: x["metrics"].get("val_loss", float('inf')),
            default=None,
        )
        
        return best

    def get_checkpoint(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Get checkpoint for a specific epoch.
        
        Args:
            epoch: Epoch number to find.
        
        Returns:
            Checkpoint entry or None.
        """
        for c in self.checkpoints:
            if c["epoch"] == epoch:
                return c
        
        return None

    def save_metadata(self, filepath: str) -> None:
        """Save checkpoint metadata to JSON file.
        
        Args:
            filepath: Path to save the metadata file.
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump({
                "checkpoints": self.checkpoints,
                "total_checkpoints": len(self.checkpoints),
            }, f, indent=2)

    @classmethod
    def load_metadata(cls, filepath: str) -> "CheckpointMetadata":
        """Load checkpoint metadata from JSON file.
        
        Args:
            filepath: Path to the metadata file.
        
        Returns:
            Loaded CheckpointMetadata instance.
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = cls()
        metadata.checkpoints = data.get("checkpoints", [])
        
        return metadata
