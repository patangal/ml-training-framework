"""
Main training engine for the ML Framework.

The Trainer class orchestrates the entire training process, coordinating between:
- The backend (PyTorch/TensorFlow)
- Model definitions
- Data loaders
- Callbacks
- Evaluation metrics
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import numpy as np
import warnings


class Trainer:
    """Main training engine for neural network models.
    
    The Trainer class provides a unified interface for training models regardless of
    the underlying backend (PyTorch or TensorFlow). It handles the training loop,
    callback management, and result tracking.
    
    Args:
        model: The neural network model to train.
        backend: The backend adapter to use (PyTorchBackend or TensorFlowBackend).
        optimizer: Optimizer instance for the model.
        loss_fn: Loss function for computing training loss.
        metrics: List of metric instances to track during training.
        callbacks: List of callback instances for training hooks.
    """

    def __init__(
        self,
        model: Any,
        backend: Any,
        optimizer: Any = None,
        loss_fn: Any = None,
        metrics: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        self.model = model
        self.backend = backend
        self.optimizer = optimizer
        self.loss_fn = loss_fn or self._default_loss()
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        
        # Training state
        self.training_history: Dict[str, List[Any]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.current_epoch = 0
        self.is_training = False

    def _default_loss(self) -> Any:
        """Return the default loss function based on backend."""
        if isinstance(self.backend, PyTorchBackend):
            import torch.nn as nn
            return nn.CrossEntropyLoss()
        elif isinstance(self.backend, TensorFlowBackend):
            import tensorflow as tf
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            # Generic default
            from .loss_functions import CrossEntropyLoss
            return CrossEntropyLoss()

    def compile(
        self,
        optimizer: Any = None,
        loss_fn: Any = None,
        metrics: Optional[List[Any]] = None,
    ):
        """Configure the trainer with optimizer, loss function, and metrics.
        
        Args:
            optimizer: Optimizer instance for the model.
            loss_fn: Loss function for computing training loss.
            metrics: List of metric instances to track during training.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        if loss_fn is not None:
            self.loss_fn = loss_fn
        if metrics is not None:
            self.metrics = metrics

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[Any, Any]] = None,
        shuffle: bool = True,
        verbose: int = 1,
    ) -> Dict[str, List[Any]]:
        """Train the model for a fixed number of epochs.
        
        Args:
            X_train: Training input data.
            y_train: Training labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_data: Tuple (X_val, y_val) for validation.
            shuffle: Whether to shuffle the training data each epoch.
            verbose: Verbosity level (0 = silent, 1 = progress bar, 2 = summary).
        
        Returns:
            Dictionary containing training history with loss and metrics per epoch.
        
        Raises:
            StopIteration: If EarlyStopping callback triggers.
        """
        self.is_training = True
        self.current_epoch = 0
        
        # Prepare callbacks
        if not isinstance(self.callbacks, CallbackList):
            self.callbacks = CallbackList(self.callbacks)
        
        # Set model reference in callbacks for checkpointing
        for callback in self.callbacks:
            if hasattr(callback, 'model'):
                callback.model = self.model
        
        # Initialize training
        logs = {}
        try:
            self.callbacks.on_train_begin(logs)
            
            # Prepare data loader
            train_loader = self.backend.create_dataloader(
                X_train, y_train, batch_size=batch_size, shuffle=shuffle
            )
            
            val_loader = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loader = self.backend.create_dataloader(
                    X_val, y_val, batch_size=batch_size, shuffle=False
                )
            
            total_batches = len(train_loader)
            
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # Epoch begin callback
                logs = {}
                self.callbacks.on_epoch_begin(epoch, logs)
                
                # Training loop
                train_logs = self._train_epoch(
                    train_loader, batch_size=batch_size, verbose=verbose
                )
                logs.update(train_logs)
                
                # Validation loop
                if val_loader is not None:
                    val_logs = self._validate(val_loader, batch_size=batch_size)
                    logs.update(val_logs)
                
                # Update history
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        if key not in self.training_history:
                            self.training_history[key] = []
                        self.training_history[key].append(value)
                
                # Epoch end callback
                logs["total_batches"] = total_batches
                self.callbacks.on_epoch_end(epoch, logs)
            
            print("\nTraining completed successfully!")
            
        except StopIteration:
            print("Training stopped early due to EarlyStopping callback.")
        
        finally:
            self.is_training = False
            self.callbacks.on_train_end(logs)
        
        return self.get_history()

    def _train_epoch(
        self, 
        train_loader: Any, 
        batch_size: int,
        verbose: int = 1
    ) -> Dict[str, float]:
        """Execute a single training epoch.
        
        Args:
            train_loader: DataLoader for training data.
            batch_size: Batch size (redundant but kept for compatibility).
            verbose: Verbosity level.
        
        Returns:
            Dictionary containing average loss and metrics for the epoch.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Batch begin callback
            logs = {"batch": batch_idx}
            self.callbacks.on_batch_begin(batch_idx, logs)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.loss_fn(predictions, y_batch)
            
            # Backward pass and optimization
            if hasattr(self.optimizer, 'zero_grad'):
                self.optimizer.zero_grad()
            elif hasattr(self.optimizer, 'reset'):
                self.optimizer.reset()
            
            loss.backward()
            
            if hasattr(self.optimizer, 'step'):
                self.optimizer.step()
            elif hasattr(self.optimizer, 'apply_gradients'):
                gradients = self.model.compute_gradients()
                self.optimizer.apply_gradients(gradients)
            
            # Accumulate loss
            total_loss += float(loss.item()) if hasattr(loss, 'item') else float(loss)
            num_batches += 1
            
            # Batch end callback
            logs["loss"] = float(loss.item()) if hasattr(loss, 'item') else float(loss)
            self.callbacks.on_batch_end(batch_idx, logs)
        
        # Calculate average loss and metrics
        avg_loss = total_loss / num_batches
        
        result = {"loss": avg_loss}
        
        # Compute metrics on training data (optional)
        if len(self.metrics) > 0:
            predictions_all = []
            labels_all = []
            
            for X_batch, y_batch in train_loader:
                preds = self.model(X_batch)
                predictions_all.append(preds.cpu().numpy() if hasattr(preds, 'cpu') else preds)
                labels_all.append(y_batch.cpu().numpy() if hasattr(y_batch, 'cpu') else y_batch)
            
            for metric in self.metrics:
                try:
                    value = metric.compute(predictions_all, labels_all)
                    result[metric.name] = float(value)
                except Exception as e:
                    warnings.warn(f"Error computing metric {metric.name}: {e}")
        
        return result

    def _validate(
        self, 
        val_loader: Any, 
        batch_size: int
    ) -> Dict[str, float]:
        """Execute validation on the validation data.
        
        Args:
            val_loader: DataLoader for validation data.
            batch_size: Batch size (redundant but kept for compatibility).
        
        Returns:
            Dictionary containing average loss and metrics for validation.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad() if hasattr(torch, 'no_grad') else contextlib.nullcontext():
            for X_batch, y_batch in val_loader:
                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                
                total_loss += float(loss.item()) if hasattr(loss, 'item') else float(loss)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        result = {"val_loss": avg_loss}
        
        # Compute validation metrics
        if len(self.metrics) > 0:
            predictions_all = []
            labels_all = []
            
            for X_batch, y_batch in val_loader:
                preds = self.model(X_batch)
                predictions_all.append(preds.cpu().numpy() if hasattr(preds, 'cpu') else preds)
                labels_all.append(y_batch.cpu().numpy() if hasattr(y_batch, 'cpu') else y_batch)
            
            for metric in self.metrics:
                try:
                    value = metric.compute(predictions_all, labels_all)
                    result[f"val_{metric.name}"] = float(value)
                except Exception as e:
                    warnings.warn(f"Error computing validation metric {metric.name}: {e}")
        
        return result

    def evaluate(
        self, 
        X_test: Any, 
        y_test: Any, 
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test input data.
            y_test: Test labels.
            batch_size: Batch size for evaluation.
            verbose: Verbosity level.
        
        Returns:
            Dictionary containing loss and metrics for the test set.
        """
        self.model.eval()
        
        test_loader = self.backend.create_dataloader(
            X_test, y_test, batch_size=batch_size, shuffle=False
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad() if hasattr(torch, 'no_grad') else contextlib.nullcontext():
            for X_batch, y_batch in test_loader:
                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                
                total_loss += float(loss.item()) if hasattr(loss, 'item') else float(loss)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        result = {"test_loss": avg_loss}
        
        # Compute test metrics
        for metric in self.metrics:
            try:
                predictions_all = []
                labels_all = []
                
                for X_batch, y_batch in test_loader:
                    preds = self.model(X_batch)
                    predictions_all.append(preds.cpu().numpy() if hasattr(preds, 'cpu') else preds)
                    labels_all.append(y_batch.cpu().numpy() if hasattr(y_batch, 'cpu') else y_batch)
                
                value = metric.compute(predictions_all, labels_all)
                result[f"test_{metric.name}"] = float(value)
            except Exception as e:
                warnings.warn(f"Error computing test metric {metric.name}: {e}")
        
        return result

    def predict(self, X: Any, batch_size: int = 32) -> Any:
        """Generate predictions for input data.
        
        Args:
            X: Input data to make predictions on.
            batch_size: Batch size for prediction.
        
        Returns:
            Model predictions.
        """
        self.model.eval()
        
        test_loader = self.backend.create_dataloader(
            X, None, batch_size=batch_size, shuffle=False
        )
        
        all_predictions = []
        
        with torch.no_grad() if hasattr(torch, 'no_grad') else contextlib.nullcontext():
            for X_batch in test_loader:
                predictions = self.model(X_batch)
                all_predictions.append(
                    predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
                )
        
        # Concatenate all predictions
        return np.concatenate(all_predictions, axis=0)

    def get_history(self) -> Dict[str, List[Any]]:
        """Get the training history.
        
        Returns:
            Dictionary containing lists of loss and metrics per epoch.
        """
        return self.training_history.copy()

    def reset_history(self):
        """Reset the training history."""
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }


# Import statements at module level for convenience
import torch
import contextlib

from .callbacks import Callback, CallbackList, EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoardCallback
