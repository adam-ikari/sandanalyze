"""CNN-based grain detection enhancement module.

Uses a simple CNN to classify detected regions as grain or non-grain.
"""

import numpy as np
import cv2

# Try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN enhancement disabled.")


def create_cnn_model(input_shape=(64, 64, 1)):
    """Create a simple CNN model for grain classification.

    Args:
        input_shape: Input shape of the model.

    Returns:
        Compiled CNN model.
    """
    if not TF_AVAILABLE:
        return None

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def extract_roi(image, contour, size=(64, 64)):
    """Extract ROI from image based on contour.

    Args:
        image: Input image.
        contour: Contour defining the region.
        size: Target size for the ROI.

    Returns:
        Resized ROI image.
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Extract ROI
    roi = image[y:y+h, x:x+w]

    # Resize to target size
    roi = cv2.resize(roi, size)

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Normalize
    roi = roi.astype(np.float32) / 255.0

    return roi


def train_cnn_model(model, training_images, training_labels, epochs=10):
    """Train the CNN model.

    Args:
        model: CNN model to train.
        training_images: Training images.
        training_labels: Training labels.
        epochs: Number of training epochs.

    Returns:
        Trained model.
    """
    if model is None or not TF_AVAILABLE:
        return None

    # Convert to numpy arrays
    training_images = np.array(training_images)
    training_labels = np.array(training_labels)

    # Reshape for CNN
    training_images = training_images.reshape(-1, 64, 64, 1)

    # Train model
    model.fit(training_images, training_labels, epochs=epochs, verbose=0)

    return model


def predict_with_cnn(model, image, contour):
    """Predict if a region is a grain using CNN.

    Args:
        model: Trained CNN model.
        image: Input image.
        contour: Contour defining the region.

    Returns:
        Prediction probability (0-1).
    """
    if model is None or not TF_AVAILABLE:
        return 1.0  # Default to grain if CNN not available

    # Extract ROI
    roi = extract_roi(image, contour)

    # Reshape for CNN
    roi = roi.reshape(1, 64, 64, 1)

    # Predict
    prediction = model.predict(roi, verbose=0)

    return prediction[0][0]


def filter_grains_with_cnn(grains, image, model=None, threshold=0.5):
    """Filter grains using CNN.

    Args:
        grains: List of grain contours.
        image: Input image.
        model: Trained CNN model (optional).
        threshold: Classification threshold.

    Returns:
        Filtered list of grains.
    """
    if model is None or not TF_AVAILABLE:
        return grains

    filtered_grains = []

    for grain in grains:
        # Predict
        probability = predict_with_cnn(model, image, grain)

        # Keep if probability > threshold
        if probability > threshold:
            filtered_grains.append(grain)

    return filtered_grains
