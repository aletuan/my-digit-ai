"""
Evaluation script for MyDigitAI neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_mnist_data
from model import create_mnist_model
import os


def plot_confusion_matrix(y_true, y_pred, save_path='images/confusion_matrix.png'):
    """Plot confusion matrix"""
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(X_test, y_test, y_pred, model, n_samples=10):
    """Plot sample predictions"""
    # Get random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Reshape image to 28x28
        img = X_test[idx].reshape(28, 28)
        
        # Get prediction probabilities
        probs = model.predict_proba(X_test[idx:idx+1].T)
        predicted_digit = y_pred[idx]
        true_digit = np.argmax(y_test[idx])
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_digit}\nPred: {predicted_digit}\nConf: {probs[predicted_digit, 0]:.3f}')
        axes[i].axis('off')
        
        # Color code based on correctness
        if predicted_digit == true_digit:
            axes[i].set_facecolor('lightgreen')
        else:
            axes[i].set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('images/sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test.T)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for digit in range(10):
        mask = y_true == digit
        if np.sum(mask) > 0:
            class_accuracy[digit] = np.mean(y_pred[mask] == y_true[mask])
    
    # Print results
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    print(f"\nPer-class Accuracy:")
    for digit in range(10):
        acc = class_accuracy.get(digit, 0)
        print(f"  Digit {digit}: {acc:.4f}")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
    
    return y_pred, accuracy, class_accuracy


def analyze_errors(X_test, y_test, y_pred, model, n_errors=10):
    """Analyze prediction errors"""
    y_true = np.argmax(y_test, axis=1)
    error_indices = np.where(y_pred != y_true)[0]
    
    if len(error_indices) == 0:
        print("No prediction errors found!")
        return
    
    print(f"\nAnalyzing {min(n_errors, len(error_indices))} prediction errors...")
    
    # Sample some errors
    sample_errors = np.random.choice(error_indices, min(n_errors, len(error_indices)), replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(sample_errors):
        img = X_test[idx].reshape(28, 28)
        predicted_digit = y_pred[idx]
        true_digit = y_true[idx]
        
        # Get prediction probabilities
        probs = model.predict_proba(X_test[idx:idx+1].T)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_digit}\nPred: {predicted_digit}\nConf: {probs[predicted_digit, 0]:.3f}')
        axes[i].axis('off')
        axes[i].set_facecolor('lightcoral')
    
    plt.tight_layout()
    plt.savefig('images/prediction_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main evaluation function"""
    print("=" * 50)
    print("MyDigitAI - Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('data/saved_weights.npz'):
        print("Error: No trained model found!")
        print("Please run train.py first to train the model.")
        return
    
    # Load data
    print("Loading MNIST test data...")
    data = load_mnist_data(normalize=True, one_hot=True, val_size=0.2)
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Test set shape: {X_test.shape}")
    
    # Create and load model
    print("\nLoading trained model...")
    model = create_mnist_model()
    model.load_weights('data/saved_weights.npz')
    print(f"Model loaded successfully!")
    print(f"Architecture: {model.layer_sizes}")
    
    # Evaluate model
    y_pred, accuracy, class_accuracy = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    y_true = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_true, y_pred)
    
    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_sample_predictions(X_test, y_test, y_pred, model)
    
    # Analyze errors
    print("\nAnalyzing prediction errors...")
    analyze_errors(X_test, y_test, y_pred, model)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Test Samples: {len(X_test)}")
    print(f"Correct Predictions: {np.sum(y_pred == y_true)}")
    print(f"Incorrect Predictions: {np.sum(y_pred != y_true)}")
    
    # Save evaluation results
    results = {
        'test_accuracy': accuracy,
        'total_samples': len(X_test),
        'correct_predictions': int(np.sum(y_pred == y_true)),
        'incorrect_predictions': int(np.sum(y_pred != y_true)),
        'class_accuracy': class_accuracy
    }
    
    print(f"\nEvaluation Results:")
    for key, value in results.items():
        if key != 'class_accuracy':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 