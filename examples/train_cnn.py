"""
Example: Training a CNN for image classification using ML Framework.

This example demonstrates how to use the ML Framework to train a Convolutional Neural Network (CNN)
for image classification tasks. It covers:
- Model configuration and creation
- Data preparation
- Training with callbacks
- Evaluation and metrics
"""

import numpy as np
from ml_framework import (
    Trainer,
    PyTorchBackend,
    CNN,
    CNNConfig,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)


def create_synthetic_dataset(num_samples=10000, img_size=32):
    """Create a synthetic dataset for demonstration purposes.
    
    Args:
        num_samples: Number of samples to generate.
        img_size: Size of images (img_size x img_size).
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    np.random.seed(42)
    
    # Generate random images with different patterns for each class
    num_classes = 10
    
    def generate_image(class_id):
        img = np.zeros((img_size, img_size, 3))
        
        if class_id == 0:
            # Horizontal lines
            for i in range(0, img_size, 4):
                img[i:i+2, :, :] = [1, 0, 0]
        elif class_id == 1:
            # Vertical lines
            for i in range(0, img_size, 4):
                img[:, i:i+2, :] = [0, 1, 0]
        elif class_id == 2:
            # Diagonal pattern
            for i in range(img_size):
                img[i, i:i+2, :] = [0, 0, 1]
        elif class_id == 3:
            # Center square
            center = img_size // 2
            img[center-5:center+5, center-5:center+5, :] = [1, 1, 0]
        elif class_id == 4:
            # Border
            img[:2, :, :] = [1, 0, 1]
            img[-2:, :, :] = [1, 0, 1]
            img[:, :2, :] = [1, 0, 1]
            img[:, -2:, :] = [1, 0, 1]
        else:
            # Random noise with class-specific seed
            np.random.seed(class_id)
            img = np.random.rand(img_size, img_size, 3) * 0.5 + class_id / num_classes
        
        return img
    
    X = np.array([generate_image(np.random.randint(num_classes)) for _ in range(num_samples)])
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
    """Main function to demonstrate CNN training."""
    print("=" * 60)
    print("ML Framework - CNN Training Example")
    print("=" * 60)
    
    # Step 1: Create synthetic dataset
    print("\n[1/5] Creating synthetic dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_synthetic_dataset(
        num_samples=8000, img_size=32
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Image shape: {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]}")
    
    # Step 2: Configure and create model
    print("\n[2/5] Creating CNN model...")
    config = CNNConfig(
        input_shape=(32, 32, 3),
        num_classes=10,
        num_filters=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        dropout_rates=[0.5, 0.5, 0.5],
        use_batch_norm=True,
    )
    
    model = CNN(config)
    print(f"  Model created: {model.__class__.__name__}")
    
    # Step 3: Set up backend and trainer
    print("\n[3/5] Setting up training...")
    backend = PyTorchBackend(device="auto", mixed_precision=False)
    
    optimizer = backend.create_optimizer(
        model=model,
        lr=0.001,
        weight_decay=1e-4,
        optimizer_type="adam"
    )
    
    loss_fn = backend.create_loss_fn(loss_type="cross_entropy")
    
    # Define metrics
    metrics = [Accuracy(), Precision(average="macro"), Recall(average="macro"), F1Score(average="macro")]
    
    trainer = Trainer(model, backend, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)
    
    # Step 4: Configure callbacks
    print("\n[4/5] Configuring callbacks...")
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath="checkpoints/best_model_{epoch}_{val_accuracy:.2f}.pt",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    lr_scheduler = LearningRateScheduler(
        initial_lr=0.001,
        min_lr=1e-6,
        warmup_epochs=5,
        max_epochs=100
    )
    
    callbacks = [early_stopping, model_checkpoint, lr_scheduler]
    
    # Step 5: Train the model
    print("\n[5/5] Training CNN...")
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
