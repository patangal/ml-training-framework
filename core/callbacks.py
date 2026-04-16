"""
Callback system for the ML Framework.

Callbacks provide a way to hook into different stages of the training process.
They enable features like early stopping, learning rate scheduling, model checkpointing, and logging.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import time
import warnings


class Callback(ABC):
    """Base class for all callbacks.
    
    A callback is a set of methods that are called at specific points during training.
    Subclasses should override the relevant methods to implement custom behavior.
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch.
        
        Args:
            epoch: The current epoch number (0-indexed).
            logs: Dictionary containing training metrics for this epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch.
        
        Args:
            epoch: The current epoch number (0-indexed).
            logs: Dictionary containing training metrics for this epoch.
        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each batch.
        
        Args:
            batch: The current batch number (0-indexed).
            logs: Dictionary containing metrics for this batch.
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each batch.
        
        Args:
            batch: The current batch number (0-indexed).
            logs: Dictionary containing metrics for this batch.
        """
        pass

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Alias for on_batch_begin."""
        self.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Alias for on_batch_end."""
        self.on_batch_end(batch, logs)


class CallbackList(Callback):
    """Container class that holds a list of callbacks.
    
    This class allows managing multiple callbacks and calling them in sequence.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize the callback list.
        
        Args:
            callbacks: List of Callback instances to add initially.
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add a callback to the list.
        
        Args:
            callback: The callback to add.
        """
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """Callback that stops training when a monitored metric has stopped improving.
    
    This callback helps prevent overfitting by stopping training early if the model
    performance on validation data doesn't improve after a certain number of epochs.
    
    Args:
        monitor: Name of the metric to monitor (e.g., 'val_loss', 'accuracy').
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        patience: Number of epochs with no improvement before stopping training.
        mode: One of 'auto', 'min', or 'max'.
            - 'min': Lower is better (e.g., loss).
            - 'max': Higher is better (e.g., accuracy).
            - 'auto': Automatically determined based on the metric name.
        restore_best_weights: Whether to restore model weights from the epoch with best value.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.001,
        patience: int = 5,
        mode: str = "auto",
        restore_best_weights: bool = True,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        # Determine mode automatically if needed
        if mode == "auto":
            if "loss" in monitor.lower() or "error" in monitor.lower():
                self.mode = "min"
            else:
                self.mode = "max"
        else:
            self.mode = mode
        
        self.best_weights = None
        self.best_score = None
        self.epochs_since_improvement = 0
        self.epochs_without_improvement = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            return
        
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            warnings.warn(
                f"EarlyStopping monitor '{self.monitor}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )
            return
        
        # Determine if this is an improvement
        if self.mode == "min":
            improved = current_score < (self.best_score or float("inf")) - self.min_delta
        else:  # max
            improved = current_score > (self.best_score or float("-inf")) + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.epochs_since_improvement = 0
            self.epochs_without_improvement.append(epoch)
            
            # Store best weights (if available from model)
            if hasattr(self, 'model') and self.model is not None:
                self.best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            self.epochs_since_improvement += 1
        
        # Check if we should stop
        if self.epochs_since_improvement >= self.patience:
            print(f"\nEarlyStopping triggered at epoch {epoch + 1}")
            print(f"Best score: {self.best_score:.4f} (epoch {self._get_best_epoch()})")
            
            # Restore best weights if available
            if self.restore_best_weights and self.best_weights is not None:
                print("Restoring best model weights...")
                if hasattr(self, 'model') and self.model is not None:
                    self.model.load_state_dict(self.best_weights)
            
            raise StopIteration()

    def _get_best_epoch(self) -> int:
        """Get the epoch number with the best score."""
        return max(self.epochs_without_improvement) if self.epochs_without_improvement else 0


class LearningRateScheduler(Callback):
    """Callback to update learning rate based on epoch progress.
    
    Supports various scheduling strategies including:
    - Step decay: Reduces LR by a factor every N epochs
    - Exponential decay: Reduces LR exponentially
    - Cosine annealing: Follows cosine curve from max to min LR
    - Warmup: Gradually increases LR at the beginning
    
    Args:
        lr_schedule: Function that takes epoch as input and returns learning rate.
            Alternatively, can be a string for built-in schedules.
        initial_lr: Initial learning rate (default: 0.001).
        min_lr: Minimum learning rate (default: 1e-7).
        warmup_epochs: Number of epochs for warmup phase (default: 0).
    """

    SCHEDULES = {
        "step": lambda epoch, initial_lr, gamma=0.1, step_size=10: 
            initial_lr * (gamma ** (epoch // step_size)),
        "exponential": lambda epoch, initial_lr, decay_rate=0.95:
            initial_lr * (decay_rate ** epoch),
        "cosine_annealing": lambda epoch, initial_lr, max_epochs, min_lr=1e-7:
            min_lr + 0.5 * (initial_lr - min_lr) * 
            (1 + (2.0 * epoch / max_epochs)) if epoch < max_epochs else min_lr,
        "warmup_cosine": lambda epoch, initial_lr, warmup_epochs, max_epochs, min_lr=1e-7:
            self._warmup_cosine(epoch, initial_lr, warmup_epochs, max_epochs, min_lr),
    }

    def __init__(
        self,
        lr_schedule=None,
        initial_lr: float = 0.001,
        min_lr: float = 1e-7,
        warmup_epochs: int = 0,
        max_epochs: Optional[int] = None,
    ):
        self.lr_schedule = lr_schedule
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Store current learning rate
        self.current_lr = initial_lr

    @staticmethod
    def _warmup_cosine(epoch, initial_lr, warmup_epochs, max_epochs, min_lr):
        """Warmup followed by cosine annealing schedule."""
        if epoch < warmup_epochs:
            # Linear warmup
            return min_lr + (initial_lr - min_lr) * (epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            adjusted_epoch = epoch - warmup_epochs
            total_adjusted = max_epochs - warmup_epochs
            if total_adjusted <= 0:
                return initial_lr
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + 
                (2.0 * adjusted_epoch / total_adjusted))

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Calculate and set the learning rate for this epoch."""
        if self.lr_schedule is None:
            return
        
        # Handle string-based schedules
        if isinstance(self.lr_schedule, str):
            schedule_func = self.SCHEDULES.get(self.lr_schedule)
            if schedule_func is None:
                raise ValueError(f"Unknown schedule type: {self.lr_schedule}")
            
            if self.lr_schedule == "warmup_cosine":
                if self.max_epochs is None:
                    raise ValueError("max_epochs required for warmup_cosine schedule")
                lr = self._warmup_cosine(epoch, self.initial_lr, 
                                        self.warmup_epochs, self.max_epochs, self.min_lr)
            else:
                lr = schedule_func(epoch, self.initial_lr)
        elif callable(self.lr_schedule):
            # Custom function
            if self.lr_schedule == "cosine_annealing" and self.max_epochs is not None:
                lr = self.SCHEDULES["cosine_annealing"](epoch, self.initial_lr, 
                                                       self.max_epochs, self.min_lr)
            else:
                lr = self.lr_schedule(epoch)
        else:
            return
        
        # Clamp to min_lr
        self.current_lr = max(lr, self.min_lr)

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self.current_lr


class ModelCheckpoint(Callback):
    """Callback that saves model checkpoints during training.
    
    Saves the model at regular intervals or when a monitored metric improves.
    
    Args:
        filepath: Path to save the checkpoint (use {epoch} and {metric} placeholders).
        monitor: Metric name to monitor for best model saving.
        save_best_only: If True, only save the model when monitored metric improves.
        mode: 'min', 'max', or 'auto' depending on whether lower is better.
        save_weights_only: If True, save only model weights instead of full model.
        verbose: Verbosity level (0 = silent, 1 = display messages).
    """

    def __init__(
        self,
        filepath: str = "checkpoint_{epoch}_{val_loss:.2f}.pt",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "auto",
        save_weights_only: bool = False,
        verbose: int = 1,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        if mode == "auto":
            if "loss" in monitor.lower() or "error" in monitor.lower():
                self.mode = "min"
            else:
                self.mode = "max"
        else:
            self.mode = mode
        
        self.save_weights_only = save_weights_only
        self.best_score = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            return
        
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            warnings.warn(
                f"ModelCheckpoint monitor '{self.monitor}' not found in logs."
            )
            return
        
        # Determine if this is an improvement
        should_save = False
        if self.save_best_only:
            if self.mode == "min":
                improved = current_score < (self.best_score or float("inf"))
            else:
                improved = current_score > (self.best_score or float("-inf"))
            
            if improved:
                self.best_score = current_score
                should_save = True
        else:
            # Save every epoch
            should_save = True
        
        if should_save and hasattr(self, 'model') and self.model is not None:
            filepath = self.filepath.format(epoch=epoch + 1, val_loss=current_score)
            
            try:
                if self.save_weights_only:
                    self.model.save_state_dict(filepath)
                else:
                    self.model.save(filepath)
                
                if self.verbose > 0:
                    status = "best" if self.save_best_only else "epoch"
                    print(f"\n{status.capitalize()} model saved to {filepath}")
            except Exception as e:
                warnings.warn(f"Failed to save checkpoint: {e}")


class TensorBoardCallback(Callback):
    """Callback that logs training metrics to TensorBoard.
    
    Useful for visualizing training progress, loss curves, and other metrics.
    
    Args:
        log_dir: Directory where logs will be written (default: "logs").
        write_graph: Whether to write the model graph to TensorBoard.
        histogram_freq: Frequency of histograms logging (0 = disabled).
    """

    def __init__(
        self,
        log_dir: str = "logs",
        write_graph: bool = True,
        histogram_freq: int = 0,
    ):
        self.log_dir = log_dir
        self.write_graph = write_graph
        self.histogram_freq = histogram_freq
        
        # Lazy import to avoid dependency on tensorboard unless used
        try:
            from torch.utils import tensorboard
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)
            self.has_tensorboard = True
        except ImportError:
            print("TensorBoard not available. Install with 'pip install tensorboard'")
            self.writer = None
            self.has_tensorboard = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.has_tensorboard or self.writer is None:
            return
        
        if logs is None:
            return
        
        for metric_name, value in logs.items():
            try:
                self.writer.add_scalar(metric_name, float(value), epoch)
            except (ValueError, TypeError):
                pass  # Skip non-numeric values

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        if not self.has_tensorboard or self.writer is None:
            return
        
        # Log model graph if available and write_graph is enabled
        if hasattr(self, 'model') and self.model is not None and self.write_graph:
            try:
                dummy_input = next(iter(self.model.dummy_inputs))
                self.writer.add_graph(self.model, dummy_input)
            except Exception as e:
                print(f"Could not log model graph: {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        if self.has_tensorboard and self.writer is not None:
            self.writer.close()


class ProgressBarCallback(Callback):
    """Callback that displays a progress bar during training.
    
    Args:
        total_epochs: Total number of epochs for the progress bar.
        batch_size: Batch size for calculating progress.
        verbose: Verbosity level (0 = disabled, 1 = enabled).
    """

    def __init__(self, total_epochs: int, batch_size: int, verbose: int = 1):
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.current_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        if self.verbose > 0:
            print(f"Training for {self.total_epochs} epochs...")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        self.current_epoch = epoch + 1
        
        if self.verbose > 0:
            # Clear previous line and show progress bar
            print(f"\nEpoch {self.current_epoch}/{self.total_epochs}")

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        if self.verbose <= 0 or logs is None:
            return
        
        total_batches = logs.get("total_batches", 1)
        progress = (batch + 1) / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        # Build progress bar string
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        percent = progress * 100
        
        # Get loss value if available
        loss_val = logs.get("loss", 0.0)
        if isinstance(loss_val, float):
            print(f"\r[{bar}] {percent:.1f}% | Loss: {loss_val:.4f}", end="")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.verbose > 0 and logs is not None:
            print()  # Newline after progress bar
            
            # Display epoch summary
            metrics_to_show = ["loss", "accuracy", "val_loss", "val_accuracy"]
            for metric in metrics_to_show:
                if metric in logs:
                    value = logs[metric]
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
