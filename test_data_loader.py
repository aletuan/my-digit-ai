"""
Test script for data_loader.py functions
Tests input/output for each function to ensure correct behavior
"""

import numpy as np
import sys
import os

# Add current directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MNISTDataLoader, load_mnist_data


def test_mnist_data_loader_init():
    """Test MNISTDataLoader initialization"""
    print("=" * 50)
    print("Testing MNISTDataLoader.__init__()")
    print("=" * 50)
    
    # Test with default data_dir
    loader = MNISTDataLoader()
    assert loader.data_dir == 'data'
    assert hasattr(loader, 'scaler')
    print("✅ __init__() with default data_dir: PASS")
    
    # Test with custom data_dir
    custom_loader = MNISTDataLoader(data_dir='test_data')
    assert custom_loader.data_dir == 'test_data'
    print("✅ __init__() with custom data_dir: PASS")
    
    # Clean up
    if os.path.exists('test_data'):
        import shutil
        shutil.rmtree('test_data')
    
    print("✅ MNISTDataLoader initialization: ALL TESTS PASSED\n")


def test_one_hot_encode():
    """Test _one_hot_encode function"""
    print("=" * 50)
    print("Testing _one_hot_encode()")
    print("=" * 50)
    
    loader = MNISTDataLoader()
    
    # Test case 1: Single label
    labels = np.array([2])
    expected = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    result = loader._one_hot_encode(labels)
    assert np.array_equal(result, expected)
    print("✅ Single label encoding: PASS")
    
    # Test case 2: Multiple labels
    labels = np.array([2, 5, 0, 7])
    expected = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]   # 7
    ])
    result = loader._one_hot_encode(labels)
    assert np.array_equal(result, expected)
    print("✅ Multiple labels encoding: PASS")
    
    # Test case 3: All digits 0-9
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = loader._one_hot_encode(labels)
    assert result.shape == (10, 10)
    assert np.sum(result) == 10  # Exactly 10 ones
    assert np.all(np.sum(result, axis=1) == 1)  # One 1 per row
    print("✅ All digits encoding: PASS")
    
    print("✅ _one_hot_encode(): ALL TESTS PASSED\n")


def test_split_train_val():
    """Test split_train_val function"""
    print("=" * 50)
    print("Testing split_train_val()")
    print("=" * 50)
    
    loader = MNISTDataLoader()
    
    # Create dummy data
    n_samples = 1000
    n_features = 784
    n_classes = 10
    
    # Test case 1: Integer labels
    X = np.random.rand(n_samples, n_features)
    y_int = np.random.randint(0, 10, n_samples)
    
    X_train, X_val, y_train, y_val = loader.split_train_val(X, y_int, val_size=0.2)
    
    assert len(X_train) == 800  # 80% of 1000
    assert len(X_val) == 200    # 20% of 1000
    assert len(y_train) == 800
    assert len(y_val) == 200
    print("✅ Integer labels split: PASS")
    
    # Test case 2: One-hot encoded labels
    y_onehot = loader._one_hot_encode(y_int)
    X_train, X_val, y_train, y_val = loader.split_train_val(X, y_onehot, val_size=0.2)
    
    assert len(X_train) == 800
    assert len(X_val) == 200
    assert y_train.shape == (800, 10)
    assert y_val.shape == (200, 10)
    print("✅ One-hot labels split: PASS")
    
    # Test case 3: Different val_size
    X_train, X_val, y_train, y_val = loader.split_train_val(X, y_int, val_size=0.3)
    assert len(X_train) == 700  # 70% of 1000
    assert len(X_val) == 300    # 30% of 1000
    print("✅ Custom val_size: PASS")
    
    print("✅ split_train_val(): ALL TESTS PASSED\n")


def test_get_batch():
    """Test get_batch function"""
    print("=" * 50)
    print("Testing get_batch()")
    print("=" * 50)
    
    loader = MNISTDataLoader()
    
    # Create dummy data
    n_samples = 100
    n_features = 784
    n_classes = 10
    
    X = np.random.rand(n_samples, n_features)
    y = loader._one_hot_encode(np.random.randint(0, 10, n_samples))
    
    # Test case 1: Default batch generation
    batch_count = 0
    total_samples = 0
    for batch_X, batch_y in loader.get_batch(X, y, batch_size=32):
        batch_count += 1
        total_samples += len(batch_X)
        assert batch_X.shape[1] == n_features
        assert batch_y.shape[1] == n_classes
        assert len(batch_X) == len(batch_y)
    
    assert total_samples == n_samples
    print(f"✅ Batch generation: {batch_count} batches, {total_samples} total samples")
    
    # Test case 2: Custom batch size
    batch_count = 0
    for batch_X, batch_y in loader.get_batch(X, y, batch_size=10):
        batch_count += 1
        assert len(batch_X) <= 10
    
    assert batch_count == 10  # 100 samples / 10 batch_size = 10 batches
    print("✅ Custom batch size: PASS")
    
    # Test case 3: No shuffle
    first_batch = None
    for batch_X, batch_y in loader.get_batch(X, y, batch_size=32, shuffle=False):
        if first_batch is None:
            first_batch = batch_X.copy()
        break
    
    # Run again and check first batch is the same
    for batch_X, batch_y in loader.get_batch(X, y, batch_size=32, shuffle=False):
        assert np.array_equal(batch_X, first_batch)
        break
    print("✅ No shuffle: PASS")
    
    print("✅ get_batch(): ALL TESTS PASSED\n")


def test_load_data():
    """Test load_data function"""
    print("=" * 50)
    print("Testing load_data()")
    print("=" * 50)
    
    loader = MNISTDataLoader()
    
    # Test case 1: With normalization and one-hot encoding
    X_train, y_train, X_test, y_test = loader.load_data(normalize=True, one_hot=True)
    
    # Check shapes
    assert X_train.shape[1] == 784  # 28x28 pixels
    assert X_test.shape[1] == 784
    assert y_train.shape[1] == 10   # 10 classes
    assert y_test.shape[1] == 10
    
    # Check normalization
    assert np.max(X_train) <= 1.0
    assert np.min(X_train) >= 0.0
    assert np.max(X_test) <= 1.0
    assert np.min(X_test) >= 0.0
    print("✅ Normalization and one-hot encoding: PASS")
    
    # Test case 2: Without normalization
    X_train, y_train, X_test, y_test = loader.load_data(normalize=False, one_hot=True)
    assert np.max(X_train) > 1.0  # Should not be normalized
    print("✅ Without normalization: PASS")
    
    # Test case 3: Without one-hot encoding
    X_train, y_train, X_test, y_test = loader.load_data(normalize=True, one_hot=False)
    assert y_train.ndim == 1  # Should be 1D array
    assert y_test.ndim == 1
    print("✅ Without one-hot encoding: PASS")
    
    print("✅ load_data(): ALL TESTS PASSED\n")


def test_load_mnist_data():
    """Test load_mnist_data convenience function"""
    print("=" * 50)
    print("Testing load_mnist_data()")
    print("=" * 50)
    
    # Test case 1: Default parameters
    data = load_mnist_data()
    
    # Check all required keys exist
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    for key in required_keys:
        assert key in data
        print(f"✅ Key '{key}' exists: PASS")
    
    # Check shapes
    assert data['X_train'].shape[1] == 784
    assert data['y_train'].shape[1] == 10
    assert data['X_val'].shape[1] == 784
    assert data['y_val'].shape[1] == 10
    assert data['X_test'].shape[1] == 784
    assert data['y_test'].shape[1] == 10
    
    # Check data splits
    total_train = len(data['X_train']) + len(data['X_val'])
    assert total_train == 60000  # Original training set size
    assert len(data['X_test']) == 10000  # Test set size
    print("✅ Data shapes and splits: PASS")
    
    # Test case 2: Custom parameters
    data_custom = load_mnist_data(normalize=False, one_hot=False, val_size=0.3)
    assert data_custom['y_train'].ndim == 1  # Integer labels
    assert data_custom['y_val'].ndim == 1
    assert data_custom['y_test'].ndim == 1
    print("✅ Custom parameters: PASS")
    
    print("✅ load_mnist_data(): ALL TESTS PASSED\n")


def test_data_consistency():
    """Test data consistency across different calls"""
    print("=" * 50)
    print("Testing Data Consistency")
    print("=" * 50)
    
    # Test that multiple calls return consistent data
    data1 = load_mnist_data()
    data2 = load_mnist_data()
    
    # Check that training data is the same
    assert np.array_equal(data1['X_train'], data2['X_train'])
    assert np.array_equal(data1['y_train'], data2['y_train'])
    print("✅ Data consistency across calls: PASS")
    
    # Test that validation split is reproducible
    data3 = load_mnist_data(val_size=0.2)
    data4 = load_mnist_data(val_size=0.2)
    assert np.array_equal(data3['X_val'], data4['X_val'])
    print("✅ Validation split reproducibility: PASS")
    
    print("✅ Data Consistency: ALL TESTS PASSED\n")


def main():
    """Run all tests"""
    print("🧪 Starting comprehensive data_loader tests...\n")
    
    try:
        test_mnist_data_loader_init()
        test_one_hot_encode()
        test_split_train_val()
        test_get_batch()
        test_load_data()
        test_load_mnist_data()
        test_data_consistency()
        
        print("🎉 ALL TESTS PASSED! Data loader is working correctly.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All data_loader functions are working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 