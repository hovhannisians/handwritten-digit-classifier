"""
Training script for the handwritten digit classifier.

This module loads the MNIST dataset, preprocesses it, defines a
Convolutional Neural Network (CNN), trains the model, evaluates it on
the test set, plots training diagnostics and optionally launches an
interactive Gradio demo.  It can be run as a standalone script:

    python train.py --epochs 10 --batch-size 64

The default hyperparameters reproduce the results described in the
README.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix


def load_and_preprocess_data():
    """Load MNIST data and preprocess it.

    Returns
    -------
    train_images : np.ndarray
        Normalised training images with shape (n_samples, 28, 28, 1).
    train_labels_onehot : np.ndarray
        One‑hot encoded training labels with shape (n_samples, 10).
    test_images : np.ndarray
        Normalised test images with shape (n_samples, 28, 28, 1).
    test_labels_onehot : np.ndarray
        One‑hot encoded test labels with shape (n_samples, 10).
    train_labels_int : np.ndarray
        Integer training labels (used for confusion matrix and misclassification analysis).
    test_labels_int : np.ndarray
        Integer test labels.
    """
    # Load the MNIST dataset from Keras.  This returns integer images
    # and labels.  Each image is 28x28 pixels and grayscale.
    (train_images, train_labels_int), (test_images, test_labels_int) = tf.keras.datasets.mnist.load_data()

    # Convert images to float32 and normalise to [0,1] as recommended【328439396556496†L520-L552】.
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Add a channel dimension (Keras expects 4‑D tensors: batch, height, width, channels).
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    # Convert integer labels to one‑hot encoded vectors.
    train_labels_onehot = tf.keras.utils.to_categorical(train_labels_int, num_classes=10)
    test_labels_onehot = tf.keras.utils.to_categorical(test_labels_int, num_classes=10)

    return (train_images, train_labels_onehot,
            test_images, test_labels_onehot,
            train_labels_int, test_labels_int)


def build_model():
    """Construct the CNN model.

    The architecture mirrors the baseline described in the README: a
    single convolutional layer, a max‑pooling layer, a dense hidden
    layer and a softmax output layer【328439396556496†L604-L643】.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_uniform',
                       input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile with SGD optimizer and categorical cross‑entropy loss【328439396556496†L604-L643】.
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_history(history):
    """Plot training and validation accuracy and loss curves.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        History object returned by model.fit().
    """
    plt.figure(figsize=(12, 4))
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels):
    """Compute and display the confusion matrix.

    Parameters
    ----------
    true_labels : np.ndarray
        Integer true labels.
    pred_labels : np.ndarray
        Integer predicted labels.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=range(10), yticklabels=range(10),
           title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.show()


def show_misclassified(images, true_labels, pred_labels, n=6):
    """Display a few misclassified test images with their true and predicted labels.

    Parameters
    ----------
    images : np.ndarray
        Test images (normalised and with channel dimension).
    true_labels : np.ndarray
        Integer true labels.
    pred_labels : np.ndarray
        Integer predicted labels.
    n : int, optional
        Number of misclassified examples to display.
    """
    mis_idx = np.where(true_labels != pred_labels)[0]
    if mis_idx.size == 0:
        print("No misclassified samples found.")
        return
    n = min(n, mis_idx.size)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for idx, ax in zip(mis_idx[:n], axes.flat):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {true_labels[idx]}, Pred: {pred_labels[idx]}')
        ax.axis('off')
    plt.show()


def launch_demo(model):
    """Launch a Gradio demo to interact with the trained model."""
    try:
        import gradio as gr  # type: ignore
    except ImportError:
        raise ImportError("gradio is not installed.  Install it with `pip install gradio` to use the demo.")

    def classify_digit(image):
        # Convert PIL image to 28x28 grayscale numpy array and normalise
        image = np.array(image.convert('L').resize((28, 28))).astype('float32') / 255.0
        image = np.expand_dims(image, (0, -1))
        pred = model.predict(image)[0]
        # Return dictionary of class probabilities for Gradio
        return {str(i): float(pred[i]) for i in range(10)}

    iface = gr.Interface(fn=classify_digit,
                         inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=False),
                         outputs=gr.outputs.Label(num_top_classes=3),
                         title="MNIST Digit Classifier",
                         description="Draw a digit (0–9) and the model will predict it.")
    iface.launch()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN for MNIST digit classification.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of training data to use for validation.')
    parser.add_argument('--demo', action='store_true', help='Launch an interactive Gradio demo after training.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and preprocess the dataset
    (train_images, train_labels_onehot,
     test_images, test_labels_onehot,
     train_labels_int, test_labels_int) = load_and_preprocess_data()

    # Build and train the model
    model = build_model()
    history = model.fit(
        train_images, train_labels_onehot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=2
    )

    # Evaluate on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels_onehot, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot training history
    plot_history(history)

    # Predict on test images
    pred_labels_int = np.argmax(model.predict(test_images), axis=1)

    # Plot confusion matrix and show misclassified examples
    plot_confusion_matrix(test_labels_int, pred_labels_int)
    show_misclassified(test_images, test_labels_int, pred_labels_int)

    # Launch demo if requested
    if args.demo:
        launch_demo(model)


if __name__ == '__main__':
    main()