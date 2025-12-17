import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/mnist_cnn.h5"
IMAGE_SIZE = (28, 28)


def load_model(model_path: str):
    """Load trained CNN model."""
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image_path: str):
    """
    Load and preprocess an image for MNIST CNN inference.
    - Convert to grayscale
    - Resize to 28x28
    - Normalize to [0,1]
    - Add batch & channel dimensions
    """
    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (28,28,1)
    image_array = np.expand_dims(image_array, axis=0)   # (1,28,28,1)

    return image_array


def predict_digit(model, image_tensor):
    """Run inference and return predicted digit + confidence."""
    predictions = model.predict(image_tensor)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_class, confidence


def main(args):
    model = load_model(MODEL_PATH)
    image_tensor = preprocess_image(args.image)

    digit, confidence = predict_digit(model, image_tensor)

    print("\nInference Result")
    print("----------------")
    print(f"Predicted digit : {digit}")
    print(f"Confidence      : {confidence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Classifier Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (handwritten digit)"
    )

    args = parser.parse_args()
    main(args)