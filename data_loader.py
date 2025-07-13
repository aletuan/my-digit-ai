"""
Data loader for MNIST dataset
Handles downloading, preprocessing, and loading of MNIST data

This module provides functionality to:
1. Load MNIST dataset using scikit-learn's fetch_openml
2. Preprocess data (normalization, one-hot encoding)
3. Split data into train/validation/test sets
4. Generate batches for training
5. Provide convenient interface for data loading

Classes:
    MNISTDataLoader: Main class for handling MNIST dataset operations

Functions:
    load_mnist_data: Convenience function for quick data loading
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MNISTDataLoader:
    """
    Handles loading and preprocessing of MNIST dataset
    
    This class provides a comprehensive interface for working with the MNIST dataset,
    including data loading, preprocessing, and batch generation for training.
    
    Attributes:
        data_dir (str): Directory to store/load data files
        scaler (StandardScaler): Scaler for data normalization (currently unused)
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the MNIST data loader
        
        Args:
            data_dir (str): Directory path for storing data files
        """
        self.data_dir = data_dir
        # Note: StandardScaler is kept for potential future use
        # Currently not used since we normalize to [0,1] range
        self.scaler = StandardScaler()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
    
    def download_mnist(self):
        """
        Load MNIST dataset using scikit-learn
        
        Note: This method is kept for compatibility but now uses scikit-learn
        instead of downloading from the original MNIST URL (which is no longer available).
        The actual data loading is done in load_data() method.
        """
        print("Loading MNIST dataset using scikit-learn...")
        # This method is kept for compatibility but now uses scikit-learn
        pass
    
    def _read_images(self, filepath):
        """
        Read image data from MNIST format (kept for compatibility)
        
        Args:
            filepath (str): Path to MNIST image file
            
        Note: This method is kept for compatibility but not used with scikit-learn.
        The actual data loading is now handled by fetch_openml in load_data().
        """
        # This method is kept for compatibility but not used with scikit-learn
        pass
    
    def _read_labels(self, filepath):
        """
        Read label data from MNIST format (kept for compatibility)
        
        Args:
            filepath (str): Path to MNIST label file
            
        Note: This method is kept for compatibility but not used with scikit-learn.
        The actual data loading is now handled by fetch_openml in load_data().
        """
        # This method is kept for compatibility but not used with scikit-learn
        pass
    
    def load_data(self, normalize=True, one_hot=True):
        """
        Load and preprocess MNIST data using scikit-learn
        
        This method loads the MNIST dataset using scikit-learn's fetch_openml,
        which automatically downloads the dataset if not already cached locally.
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0, 1] range
            one_hot (bool): Whether to one-hot encode the labels
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test) - Training and test data
            
        Note:
            - Training set: 60,000 samples
            - Test set: 10,000 samples
            - Each image is 28x28 pixels (784 features)
            - Labels are digits 0-9
        """
        print("Loading MNIST data using scikit-learn...")
        
        # Import scikit-learn's fetch_openml for MNIST
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        
        # Load MNIST dataset
        print("Fetching MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # Extract features and labels
        X, y = mnist.data, mnist.target.astype(int)
        
        # Split into train and test sets
        # Using 10,000 samples for test set (standard MNIST split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=42, stratify=y
        )
        
        # Normalize images to [0, 1] range
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
            print("Data normalized to [0, 1] range")
        
        # One-hot encode labels
        if one_hot:
            y_train = self._one_hot_encode(y_train)
            y_test = self._one_hot_encode(y_test)
            print("Labels converted to one-hot encoding")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _one_hot_encode(self, labels):
        """
        Convert labels to one-hot encoding
        
        Args:
            labels (np.ndarray): Array of integer labels (0-9)
            
        Returns:
            np.ndarray: One-hot encoded labels with shape (n_samples, 10)
            
        Example:
            Input: [2, 5, 0]
            Output: [[0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0],
                    [1,0,0,0,0,0,0,0,0,0]]
        """
        n_classes = 10  # MNIST has 10 classes (digits 0-9)
        n_samples = len(labels)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1
        return one_hot
    
    def split_train_val(self, train_images, train_labels, val_size=0.2, random_state=42):
        """
        Split training data into training and validation sets
        
        Args:
            train_images (np.ndarray): Training images
            train_labels (np.ndarray): Training labels (one-hot encoded or integer)
            val_size (float): Proportion of training data to use for validation
            random_state (int): Random seed for reproducible splits
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val) - Split training and validation data
            
        Note:
            Uses stratified sampling to ensure balanced class distribution
        """
        # Handle both one-hot encoded and integer labels for stratification
        if train_labels.ndim > 1:
            # One-hot encoded labels
            stratify_labels = np.argmax(train_labels, axis=1)
        else:
            # Integer labels
            stratify_labels = train_labels
            
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, 
            test_size=val_size, 
            random_state=random_state,
            stratify=stratify_labels
        )
        
        return X_train, X_val, y_train, y_val
    
    def get_batch(self, images, labels, batch_size, shuffle=True):
        """
        Generate batches of data for training
        
        This method creates an iterator that yields batches of data,
        which is useful for training neural networks with mini-batch gradient descent.
        
        Args:
            images (np.ndarray): Input images
            labels (np.ndarray): Corresponding labels
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data before batching
            
        Yields:
            tuple: (batch_images, batch_labels) - Batch of data
            
        Example:
            >>> loader = MNISTDataLoader()
            >>> X_train, y_train, X_test, y_test = loader.load_data()
            >>> for batch_X, batch_y in loader.get_batch(X_train, y_train, batch_size=32):
            ...     # Train model with this batch
            ...     pass
        """
        n_samples = len(images)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield images[batch_indices], labels[batch_indices]


def load_mnist_data(normalize=True, one_hot=True, val_size=0.2):
    """
    Convenience function to load MNIST data with train/validation/test split
    
    This is the main function used by other modules (train.py, evaluate.py)
    to quickly load and preprocess the MNIST dataset.
    
    Args:
        normalize (bool): Whether to normalize pixel values to [0, 1] range
        one_hot (bool): Whether to one-hot encode the labels
        val_size (float): Proportion of training data to use for validation
        
    Returns:
        dict: Dictionary containing all data splits with keys:
            - 'X_train': Training images
            - 'y_train': Training labels
            - 'X_val': Validation images  
            - 'y_val': Validation labels
            - 'X_test': Test images
            - 'y_test': Test labels
            
    Example:
        >>> data = load_mnist_data()
        >>> X_train = data['X_train']  # Shape: (48000, 784)
        >>> y_train = data['y_train']  # Shape: (48000, 10) - one-hot encoded
        >>> X_test = data['X_test']    # Shape: (10000, 784)
        >>> y_test = data['y_test']    # Shape: (10000, 10) - one-hot encoded
    """
    loader = MNISTDataLoader()
    train_images, train_labels, test_images, test_labels = loader.load_data(
        normalize=normalize, one_hot=one_hot
    )
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = loader.split_train_val(
        train_images, train_labels, val_size=val_size
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': test_images,
        'y_test': test_labels
    }


if __name__ == "__main__":
    """
    Test script for the data loader
    
    This section runs when the file is executed directly,
    providing a quick way to test the data loading functionality.
    """
    print("Testing MNIST data loader...")
    
    # Test the data loader
    data = load_mnist_data()
    print("Data loaded successfully!")
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Validation samples: {data['X_val'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    
    # Test batch generation
    loader = MNISTDataLoader()
    batch_count = 0
    for batch_X, batch_y in loader.get_batch(data['X_train'], data['y_train'], batch_size=32):
        batch_count += 1
        if batch_count == 1:
            print(f"First batch shape: X={batch_X.shape}, y={batch_y.shape}")
        if batch_count >= 3:
            break
    print(f"Batch generation test completed. Generated {batch_count} batches.") 