# MyDigitAI - Neural Network for Handwritten Digit Recognition

## Overview

MyDigitAI is a neural network-based application designed to recognize handwritten digits (0-9). The system is trained using the MNIST dataset and built from scratch using Python and NumPy, implementing all fundamental aspects of a deep learning pipeline: forward propagation, backpropagation, gradient descent optimization, and evaluation.

The final product includes a simple GUI where users can draw digits and the model will predict the number with high accuracy.

## Features

- **Custom Neural Network**: Built from scratch using only NumPy
- **MNIST Dataset**: Trained on the standard handwritten digit dataset
- **Interactive GUI**: Draw digits and get instant predictions
- **Training Visualization**: Real-time plots of loss and accuracy
- **High Accuracy**: Expected >90% accuracy on test set

## Neural Network Architecture

- **Input Layer**: 784 neurons (flattened 28x28 pixel image)
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons, Softmax activation (representing classes 0-9)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd my-digit-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

This will:
- Download the MNIST dataset
- Train the neural network
- Save the trained weights to `saved_weights.npz`
- Display training progress and plots

### Running the GUI

```bash
python gui.py
```

This opens an interactive interface where you can:
- Draw digits on the canvas
- Click "Predict" to get the model's prediction
- Clear the canvas to try again

### Evaluation

```bash
python evaluate.py
```

This evaluates the trained model on the test set and displays:
- Overall accuracy
- Confusion matrix
- Sample predictions

## Project Structure

```
mydigitai/
├── data_loader.py      # MNIST dataset loading and preprocessing
├── model.py           # Neural network implementation
├── train.py           # Training script
├── evaluate.py        # Model evaluation
├── gui.py            # Interactive GUI application
├── utils.py          # Utility functions
├── saved_weights.npz # Trained model weights
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Key Algorithms

- **Activation Functions**: ReLU, Softmax
- **Loss Function**: Cross-entropy
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Regularization**: L2 penalty (optional)

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

## Metrics

- Accuracy on MNIST test set (>90% expected)
- Loss and accuracy plots by epoch
- Confusion matrix (optional)

## Future Extensions

- Add CNN for better performance
- Export model to ONNX or TensorFlow Lite
- Add support for webcam digit input
- Train on custom digits drawn by user

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- MNIST dataset creators
- NumPy development team
- Matplotlib for visualization 