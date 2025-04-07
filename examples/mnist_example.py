"""
Example of using Nexor to train a simple CNN on MNIST dataset.
This example demonstrates the key features of Nexor including:
- Easy model creation using Sequential API
- Hybrid backend utilizing both TensorFlow and PyTorch capabilities
- Simple training interface similar to Keras
"""

import numpy as np
from nexor.core import Tensor
from nexor.nn import Sequential, Conv2D, MaxPool2D, Linear, ReLU, Dropout
import tensorflow as tf  # For loading MNIST dataset

def load_mnist():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape data
    x_train = x_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create a simple CNN model using Nexor"""
    model = Sequential([
        # Convolutional layers
        Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2),
        
        Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2),
        
        # Flatten and dense layers
        Lambda(lambda x: x.reshape(x.shape[0], -1)),  # Flatten
        Linear(in_features=64*7*7, out_features=512),
        ReLU(),
        Dropout(0.5),
        Linear(in_features=512, out_features=10)
    ])
    
    return model

def main():
    # Load data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        x=Tensor(x_train[:10000]),  # Using subset for example
        y=Tensor(y_train[:10000]),
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=True
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_predictions = model.predict(Tensor(x_test[:1000]))
    test_pred_classes = np.argmax(test_predictions.numpy(), axis=1)
    test_true_classes = np.argmax(y_test[:1000], axis=1)
    accuracy = np.mean(test_pred_classes == test_true_classes)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()