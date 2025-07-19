"""
Utility functions for MyDigitAI project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


def save_training_results(results, filename='training_results.json'):
    """Save training results to JSON file"""
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training results saved to {filename}")


def load_training_results(filename='training_results.json'):
    """Load training results from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"Results file {filename} not found.")
        return None


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path='images/learning_curves.png'):
    """Plot learning curves"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_weights(model, layer_idx=0, save_path='images/weight_visualization.png'):
    """Visualize weights of a specific layer"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    if layer_idx >= len(model.weights):
        print(f"Layer {layer_idx} does not exist. Model has {len(model.weights)} layers.")
        return
    
    weights = model.weights[layer_idx]
    
    # Reshape weights for visualization (assuming first layer)
    if layer_idx == 0:
        # For first layer, reshape to 28x28 images
        n_neurons = weights.shape[0]
        n_features = weights.shape[1]
        
        if n_features == 784:  # MNIST input size
            # Reshape to 28x28
            weights_reshaped = weights.reshape(n_neurons, 28, 28)
            
            # Plot first 16 neurons
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(min(16, n_neurons)):
                axes[i].imshow(weights_reshaped[i], cmap='RdBu_r', aspect='equal')
                axes[i].set_title(f'Neuron {i}')
                axes[i].axis('off')
            
            plt.suptitle(f'Weight Visualization - Layer {layer_idx}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"Unexpected input size: {n_features}")
    else:
        print(f"Weight visualization for layer {layer_idx} not implemented yet.")


def compute_model_statistics(model, X_test, y_test):
    """Compute detailed model statistics"""
    y_pred = model.predict(X_test.T)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Per-class accuracy
    class_accuracy = {}
    class_counts = {}
    
    for digit in range(10):
        mask = y_true == digit
        if np.sum(mask) > 0:
            class_accuracy[digit] = np.mean(y_pred[mask] == y_true[mask])
            class_counts[digit] = np.sum(mask)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Precision, Recall, F1-score
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'overall_accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'class_counts': class_counts,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def save_model_metadata(model, training_history, test_accuracy, filename='model_metadata.json'):
    """Save model metadata"""
    metadata = {
        'architecture': model.layer_sizes,
        'learning_rate': model.learning_rate,
        'l2_lambda': model.l2_lambda,
        'test_accuracy': test_accuracy,
        'training_history': {
            'final_train_loss': training_history['train_losses'][-1],
            'final_val_loss': training_history['val_losses'][-1],
            'final_train_accuracy': training_history['train_accuracies'][-1],
            'final_val_accuracy': training_history['val_accuracies'][-1],
            'epochs_trained': len(training_history['train_losses'])
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {filename}")


def create_model_summary(model, X_test, y_test):
    """Create a comprehensive model summary"""
    stats = compute_model_statistics(model, X_test, y_test)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Architecture: {model.layer_sizes}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"L2 Regularization: {model.l2_lambda}")
    print(f"Total Parameters: {sum(w.size + b.size for w, b in zip(model.weights, model.biases))}")
    print(f"Overall Test Accuracy: {stats['overall_accuracy']:.4f}")
    
    print("\nPer-class Accuracy:")
    for digit in range(10):
        acc = stats['class_accuracy'].get(digit, 0)
        count = stats['class_counts'].get(digit, 0)
        print(f"  Digit {digit}: {acc:.4f} ({count} samples)")
    
    print("\nModel Performance:")
    print(f"  Best performing digit: {max(stats['class_accuracy'], key=stats['class_accuracy'].get)}")
    print(f"  Worst performing digit: {min(stats['class_accuracy'], key=stats['class_accuracy'].get)}")
    
    return stats


def preprocess_image_for_prediction(image_path):
    """Preprocess an image file for prediction"""
    from PIL import Image
    import numpy as np
    
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0
    
    # Invert if necessary (assuming white background)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    
    return img_array.flatten()


def batch_predict(model, images, batch_size=32):
    """Make predictions on a batch of images"""
    predictions = []
    probabilities = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = np.array(batch).T  # Transpose for model input
        
        pred = model.predict(batch)
        prob = model.predict_proba(batch)
        
        predictions.extend(pred)
        probabilities.extend(prob.T)
    
    return np.array(predictions), np.array(probabilities)


if __name__ == "__main__":
    # Test utility functions
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("- save_training_results()")
    print("- load_training_results()")
    print("- plot_learning_curves()")
    print("- visualize_weights()")
    print("- compute_model_statistics()")
    print("- save_model_metadata()")
    print("- create_model_summary()")
    print("- preprocess_image_for_prediction()")
    print("- batch_predict()") 