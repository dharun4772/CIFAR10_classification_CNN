# CIFAR-10 Image Classification

## Project Overview
This project focuses on classifying images from the CIFAR-10 dataset using deep learning models. The dataset consists of 60,000 color images (32x32 pixels) categorized into 10 classes. The goal is to train and evaluate a convolutional neural network (CNN) to achieve high accuracy on image classification.

## Dataset
- **CIFAR-10** contains 60,000 images with 10 categories:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

## Model Architectures Explored
1. **Baseline CNN** – A simple model with a few convolutional layers.
2. **Deep CNN** – A deeper network with batch normalization and dropout.
3. **ResNet** – A residual network to improve performance using skip connections.
4. **VGG-like Model** – A deep architecture inspired by VGGNet.

## Preprocessing
- Normalization: Pixel values scaled to [0,1].
- Data Augmentation: Applied transformations like rotation, flipping, and zooming to improve generalization.
- One-hot encoding for labels.

## Training Details
- **Framework:** TensorFlow/Keras
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam, SGD with momentum
- **Evaluation Metric:** Accuracy
- **Batch Size:** 32/64
- **Epochs:** 50-100

## Results
- The ResNet model achieved the highest accuracy (~90%) on the test set.
- Data augmentation significantly improved generalization.
- Batch normalization accelerated training convergence.

## Future Work
- Experimenting with transformer-based vision models (e.g., ViTs).
- Fine-tuning pre-trained models for better performance.
- Implementing hyperparameter tuning using AutoML frameworks.

## Installation
To install dependencies, run:
```bash
pip install tensorflow keras matplotlib numpy
```

## Running the Code
```python
import tensorflow as tf
from tensorflow.keras import layers, models
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

## Conclusion
This project demonstrates how CNNs can effectively classify images in the CIFAR-10 dataset. Advanced architectures like ResNet outperform traditional models, showing the power of deep learning in image recognition tasks.

