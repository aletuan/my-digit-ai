# MyDigitAI - Neural Network for Handwritten Digit Recognition

## Overview

MyDigitAI is a neural network-based application designed to recognize handwritten digits (0-9). The system is trained using the MNIST dataset and built from scratch using Python and NumPy, implementing all fundamental aspects of a deep learning pipeline: forward propagation, backpropagation, gradient descent optimization, and evaluation.

The final product will include a simple GUI where users can draw digits and the model will predict the number with high accuracy.

---

## Objectives

- Build a feedforward neural network with multiple layers
- Train the network using the MNIST dataset
- Apply cross-entropy loss and gradient descent for training
- Visualize training progress and model accuracy
- Deploy a simple user interface to draw digits and get predictions

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| ML Framework | NumPy (custom NN implementation) |
| Dataset | MNIST |
| GUI | Tkinter or Web Canvas (HTML/JS) |
| Visualization | Matplotlib |

---

## Neural Network Architecture

- **Input Layer**: 784 neurons (flattened 28x28 pixel image)
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons, Softmax activation (representing classes 0-9)

---

## Training Pipeline

1. Load MNIST dataset
2. Normalize input features (0-1 range)
3. One-hot encode labels
4. Initialize weights and biases
5. Iterate through multiple epochs:
   - Forward pass to compute predictions
   - Compute loss (cross-entropy)
   - Backpropagate errors to compute gradients
   - Update weights using gradient descent
6. Evaluate on validation/test set

---

## Key Algorithms

- **Activation functions**: ReLU, Softmax
- **Loss function**: Cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Regularization**: L2 penalty (optional)

---

## User Interface (GUI)

- Canvas-based digit input
- Predict button to get result
- Display predicted number
- Option to clear and retry

---

## File Structure

```
mydigitai/
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
├── gui.py (or web/index.html)
├── utils.py
├── saved_weights.npz
└── README.md
```

---

## Metrics

- Accuracy on MNIST test set (>90% expected)
- Loss and accuracy plots by epoch
- Confusion matrix (optional)

---

## Future Extensions

- Add CNN for better performance
- Export model to ONNX or TensorFlow Lite
- Add support for webcam digit input
- Train on custom digits drawn by user

---

## License

MIT License