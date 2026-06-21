"""CNN training script for grain detection enhancement.

Usage:
    python train_cnn.py --image <image_path> --annotations <annotations_path>
"""

import argparse
import cv2
import numpy as np
from core.cnn_enhancer import create_cnn_model, train_cnn_model, extract_roi


def load_annotations(image_path, annotations_path):
    """Load annotations from file.

    Args:
        image_path: Path to the image.
        annotations_path: Path to the annotations file.

    Returns:
        Tuple of (image, contours).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Load annotations (assuming JSON format)
    import json
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    contours = []
    for annotation in annotations:
        contour = np.array(annotation['contour'], dtype=np.int32)
        contours.append(contour)

    return image, contours


def prepare_training_data(image, contours, size=(64, 64)):
    """Prepare training data from annotations.

    Args:
        image: Input image.
        contours: List of annotated contours.
        size: Target size for ROIs.

    Returns:
        Tuple of (training_images, training_labels).
    """
    training_images = []
    training_labels = []

    for contour in contours:
        # Extract ROI
        roi = extract_roi(image, contour, size)
        training_images.append(roi)
        training_labels.append(1)  # 1 = grain

    # Generate negative samples (non-grain regions)
    # For simplicity, use random regions
    for _ in range(len(contours)):
        # Random region
        x = np.random.randint(0, image.shape[1] - size[0])
        y = np.random.randint(0, image.shape[0] - size[1])
        roi = image[y:y+size[1], x:x+size[0]]
        roi = cv2.resize(roi, size)
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = roi.astype(np.float32) / 255.0
        training_images.append(roi)
        training_labels.append(0)  # 0 = non-grain

    return training_images, training_labels


def main():
    parser = argparse.ArgumentParser(description='Train CNN for grain detection')
    parser.add_argument('--image', required=True, help='Path to training image')
    parser.add_argument('--annotations', required=True, help='Path to annotations file')
    parser.add_argument('--output', default='cnn_model.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    image, contours = load_annotations(args.image, args.annotations)

    # Prepare training data
    print("Preparing training data...")
    training_images, training_labels = prepare_training_data(image, contours)

    # Create model
    print("Creating model...")
    model = create_cnn_model()

    # Train model
    print("Training model...")
    model = train_cnn_model(model, training_images, training_labels, epochs=args.epochs)

    # Save model
    print(f"Saving model to {args.output}...")
    model.save(args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()
