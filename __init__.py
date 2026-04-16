"""
ML Framework - A comprehensive machine learning model training framework
supporting neural networks with PyTorch/TensorFlow backends.

Features:
- Dual backend support (PyTorch/TensorFlow)
- Neural network architectures: CNN, RNN/LSTM, Transformer, MLP
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Hyperparameter tuning: Grid search, Random search, Bayesian optimization
- Callbacks: Early stopping, Learning rate scheduling, Model checkpointing
"""

__version__ = "1.0.0"
__author__ = "ML Framework Team"

from .core.trainer import Trainer
from .core.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoardCallback,
)
from .backends.base import BackendAdapter
from .backends.pytorch_backend import PyTorchBackend
from .backends.tensorflow_backend import TensorFlowBackend
from .models.cnn import CNN, CNNConfig
from .models.rnn import RNN, LSTM, GRU, RNNConfig
from .models.transformer import Transformer, TransformerConfig
from .models.mlp import MLP, MLPConfig
from .evaluation.metrics import (
    Metric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
)
from .evaluation.evaluator import Evaluator
from .tuning.tuner import HyperparameterTuner
from .tuning.grid_search import GridSearch
from .tuning.random_search import RandomSearch
from .tuning.bayesian import BayesianOptimization
from .persistence.checkpoint import CheckpointManager, ModelSerializer
from .config.settings import Config

__all__ = [
    # Core
    "Trainer",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "LearningRateScheduler",
    "ModelCheckpoint",
    "TensorBoardCallback",
    # Backends
    "BackendAdapter",
    "PyTorchBackend",
    "TensorFlowBackend",
    # Models
    "CNN",
    "CNNConfig",
    "RNN",
    "LSTM",
    "GRU",
    "RNNConfig",
    "Transformer",
    "TransformerConfig",
    "MLP",
    "MLPConfig",
    # Evaluation
    "Metric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "R2Score",
    "Evaluator",
    # Tuning
    "HyperparameterTuner",
    "GridSearch",
    "RandomSearch",
    "BayesianOptimization",
    # Persistence
    "CheckpointManager",
    "ModelSerializer",
    # Config
    "Config",
]
