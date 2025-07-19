"""
Neural Network Model Implementation
Custom neural network built from scratch using NumPy
"""

import numpy as np


class NeuralNetwork:
    """Feedforward neural network with multiple layers"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, l2_lambda=0.01):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            l2_lambda: L2 regularization parameter
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for better gradient flow
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Store activations and z values for backpropagation
        self.activations = []
        self.z_values = []
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data of shape (n_features, n_samples)
        
        Returns:
            Output of the final layer
        """
        self.activations = [X]
        self.z_values = []
        
        # Forward pass through all layers except the last
        for i in range(self.num_layers - 2):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)
        
        # Output layer with softmax
        z = np.dot(self.weights[-1], self.activations[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels (one-hot encoded)
        
        Returns:
            Cross-entropy loss
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[1]
        
        # Add L2 regularization
        l2_reg = 0
        for w in self.weights:
            l2_reg += np.sum(w ** 2)
        l2_reg = (self.l2_lambda / (2 * y_pred.shape[1])) * l2_reg
        
        return loss + l2_reg
    
    def backward(self, y_true):
        """
        Backward propagation
        
        Args:
            y_true: True labels (one-hot encoded)
        
        Returns:
            Gradients for weights and biases
        """
        m = y_true.shape[1]
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Error at the output layer
        delta = self.activations[-1] - y_true
        
        # Backpropagate through all layers
        for layer in range(self.num_layers - 1, 0, -1):
            # Gradient for weights and biases
            weight_gradients[layer - 1] = np.dot(delta, self.activations[layer - 1].T) / m
            bias_gradients[layer - 1] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Add L2 regularization to weight gradients
            weight_gradients[layer - 1] += (self.l2_lambda / m) * self.weights[layer - 1]
            
            # Compute delta for the previous layer (if not the first layer)
            if layer > 1:
                delta = np.dot(self.weights[layer - 1].T, delta) * self.relu_derivative(self.z_values[layer - 2])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train_step(self, X, y):
        """
        Perform one training step
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
        
        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.cross_entropy_loss(y_pred, y)
        
        # Backward pass
        weight_gradients, bias_gradients = self.backward(y)
        
        # Update parameters
        self.update_parameters(weight_gradients, bias_gradients)
        
        return loss
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
        
        Returns:
            Predicted class labels
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=0)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input data
        
        Returns:
            Prediction probabilities
        """
        return self.forward(X)
    
    def accuracy(self, X, y):
        """
        Compute accuracy
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=0)
        return np.mean(predictions == true_labels)
    
    def save_weights(self, filepath):
        """Save trained weights and biases"""
        # Save each weight and bias separately to avoid inhomogeneous shape error
        save_dict = {
            'layer_sizes': np.array(self.layer_sizes)
        }
        
        # Add weights and biases with unique names
        for i, weight in enumerate(self.weights):
            save_dict[f'weight_{i}'] = weight
        
        for i, bias in enumerate(self.biases):
            save_dict[f'bias_{i}'] = bias
        
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath):
        """Load trained weights and biases"""
        data = np.load(filepath)
        
        # Load layer sizes
        self.layer_sizes = data['layer_sizes'].tolist()
        self.num_layers = len(self.layer_sizes)
        
        # Load weights and biases
        self.weights = []
        self.biases = []
        
        # Count how many weight/bias pairs we have
        num_layers = len(self.layer_sizes) - 1
        
        for i in range(num_layers):
            self.weights.append(data[f'weight_{i}'])
            self.biases.append(data[f'bias_{i}'])
        
        print(f"Model loaded from {filepath}")


def create_mnist_model():
    """Create neural network model for MNIST dataset"""
    # Architecture: 784 -> 128 -> 64 -> 10
    layer_sizes = [784, 128, 64, 10]
    model = NeuralNetwork(layer_sizes, learning_rate=0.01, l2_lambda=0.01)
    return model


if __name__ == "__main__":
    # Test the model
    model = create_mnist_model()
    print("Neural network model created successfully!")
    print(f"Architecture: {model.layer_sizes}")
    print(f"Number of layers: {model.num_layers}") 