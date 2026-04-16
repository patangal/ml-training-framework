"""
Example: Training a Transformer model using ML Framework.

This example demonstrates how to use the ML Framework to train a Transformer model
for sequence classification tasks. It covers:
- Transformer configuration and creation
- Sequence data preparation
- Training with attention mechanisms
- Evaluation and metrics
"""

import numpy as np
from ml_framework import (
    Trainer,
    PyTorchBackend,
    Transformer,
    TransformerConfig,
    Accuracy,
    F1Score,
    EarlyStopping,
)


def create_synthetic_sequence_dataset(num_samples=5000, seq_len=64):
    """Create a synthetic sequence dataset for demonstration.
    
    Args:
        num_samples: Number of samples to generate.
        seq_len: Length of sequences.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    np.random.seed(42)
    
    # Generate random sequences with class-specific patterns
    num_classes = 5
    
    def generate_sequence(class_id):
        seq = np.zeros((seq_len, 10))  # 10-dimensional features
        
        if class_id == 0:
            # Rising pattern
            for i in range(seq_len):
                seq[i, 0] = i / seq_len
                seq[i, 1] = np.sin(i * 0.1)
        
        elif class_id == 1:
            # Falling pattern
            for i in range(seq_len):
                seq[i, 0] = (seq_len - i) / seq_len
                seq[i, 2] = np.cos(i * 0.1)
        
        elif class_id == 2:
            # Oscillating pattern
            for i in range(seq_len):
                seq[i, 3] = np.sin(i * 0.5)
                seq[i, 4] = np.cos(i * 0.5)
        
        elif class_id == 3:
            # Step function
            step = seq_len // 2
            for i in range(seq_len):
                if i < step:
                    seq[i, 5] = 0.5
                else:
                    seq[i, 5] = 1.0
        
        elif class_id == 4:
            # Random with seed
            np.random.seed(class_id)
            seq = np.random.randn(seq_len, 10) * 0.3 + class_id / num_classes
        
        return seq
    
    X = np.array([generate_sequence(np.random.randint(num_classes)) for _ in range(num_samples)])
    y = np.array([np.random.randint(num_classes) for _ in range(num_samples)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split into train/val/test
    split1 = int(0.8 * len(X))
    split2 = int(0.9 * len(X))
    
    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    """Main function to demonstrate Transformer training."""
    print("=" * 60)
    print("ML Framework - Transformer Training Example")
    print("=" * 60)
    
    # Step 1: Create synthetic dataset
    print("\n[1/5] Creating synthetic sequence dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_synthetic_sequence_dataset(
        num_samples=4000, seq_len=64
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Sequence length: {X_train.shape[1]}")
    print(f"  Feature dimension: {X_train.shape[2]}")
    
    # Step 2: Configure and create Transformer model
    print("\n[2/5] Creating Transformer model...")
    config = TransformerConfig(
        input_dim=10,
        d_model=128,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=256,
        dropout=0.1,
        max_seq_len=64,
        num_classes=5,
        is_decoder_only=False,  # Use encoder-only (BERT-style) for classification
    )
    
    model = Transformer(config)
    print(f"  Model created: {model.__class__.__name__}")
    print(f"  Parameters: ~{model.count_parameters():,} trainable parameters")
    
    # Step 3: Set up backend and trainer
    print("\n[3/5] Setting up training...")
    backend = PyTorchBackend(device="auto", mixed_precision=False)
    
    optimizer = backend.create_optimizer(
        model=model,
        lr=0.0001,  # Transformers typically use smaller learning rates
        weight_decay=1e-4,
        optimizer_type="adamw"
    )
    
    loss_fn = backend.create_loss_fn(loss_type="cross_entropy")
    
    # Define metrics
    metrics = [Accuracy(), F1Score(average="macro")]
    
    trainer = Trainer(model, backend, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)
    
    # Step 4: Configure callbacks
    print("\n[4/5] Configuring callbacks...")
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # Step 5: Train the model
    print("\n[5/5] Training Transformer...")
    print("-" * 60)
    
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=True,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    test_results = trainer.evaluate(X_test, y_test, batch_size=32, verbose=1)
    
    for metric_name, value in test_results.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
    
    # Save model summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total epochs trained: {len(history['loss'])}")
    print(f"Best validation accuracy: {max(history.get('val_accuracy', [0])):.4f}")


if __name__ == "__main__":
    main()
