"""
preprocess.py
--------------
Loads and preprocesses the MNIST dataset
for handwritten digit recognition.
"""

import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    """
    Loads MNIST dataset and preprocesses images.
    """

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values (0–255 -> 0–1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape for CNN input (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test
