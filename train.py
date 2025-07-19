"""
Training script for MyDigitAI neural network
"""

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_data
from model import create_mnist_model
import os
import time


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the neural network model
    
    Args:
        model: Neural network model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Dictionary containing training history
    """
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Batch size: {batch_size}")
    
    # Initialize history lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        batch_losses = []
        for i in range(0, X_train.shape[0], batch_size):
            batch_end = min(i + batch_size, X_train.shape[0])
            X_batch = X_train_shuffled[i:batch_end].T  # Transpose for model input
            y_batch = y_train_shuffled[i:batch_end].T  # Transpose for model input
            
            # Training step
            loss = model.train_step(X_batch, y_batch)
            batch_losses.append(loss)
        
        # Compute metrics
        avg_train_loss = np.mean(batch_losses)
        train_accuracy = model.accuracy(X_train.T, y_train.T)
        val_accuracy = model.accuracy(X_val.T, y_val.T)
        
        # Compute validation loss
        val_pred = model.forward(X_val.T)
        val_loss = model.cross_entropy_loss(val_pred, y_val.T)
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Train Acc: {train_accuracy:.4f} - "
              f"Val Acc: {val_accuracy:.4f} - "
              f"Time: {epoch_time:.2f}s")
        
        # Early stopping (optional)
        if epoch > 10 and val_losses[-1] > val_losses[-2] and val_losses[-2] > val_losses[-3]:
            print("Early stopping triggered!")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


def main():
    """Main training function"""
    print("=" * 50)
    print("MyDigitAI - Training Neural Network")
    print("=" * 50)
    
    # Load data
    print("Loading MNIST dataset...")
    data = load_mnist_data(normalize=True, one_hot=True, val_size=0.2)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Data loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create model
    print("\nCreating neural network model...")
    model = create_mnist_model()
    print(f"Model architecture: {model.layer_sizes}")
    
    # Training parameters
    epochs = 50
    batch_size = 32
    
    # Train model
    print(f"\nStarting training...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = model.accuracy(X_test.T, y_test.T)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save_weights('saved_weights.npz')
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['train_accuracies'],
        history['val_accuracies']
    )
    
    # Print final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: saved_weights.npz")
    print(f"Training plots saved to: training_history.png")
    
    # Save training results
    results = {
        'test_accuracy': test_accuracy,
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'final_train_accuracy': history['train_accuracies'][-1],
        'final_val_accuracy': history['val_accuracies'][-1],
        'epochs_trained': len(history['train_losses'])
    }
    
    print(f"\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 