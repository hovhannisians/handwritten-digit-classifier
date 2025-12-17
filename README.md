# Handwritten Digit Classifier (MNIST)

A clean, end-to-end **Convolutional Neural Network (CNN)** project for recognizing handwritten digits (0–9) using the MNIST dataset.
This repository is designed to demonstrate **core computer vision fundamentals**, **model training**, **evaluation**, and **inference** in a production-ready way.

---

## 📌 Project Overview

Handwritten digit recognition is a classic computer vision task and a common benchmark for evaluating CNN architectures.

This project focuses on:

* Building a CNN **from scratch**
* Training and evaluating on MNIST
* Analyzing model performance
* Running inference from the command line

---

## 📊 Dataset

* **MNIST** handwritten digits dataset
* 60,000 training images
* 10,000 test images
* 28×28 grayscale images
* 10 classes (digits 0–9)

The dataset is **automatically downloaded** via TensorFlow/Keras.

---

## 🧠 Model Architecture

| Layer        | Description                  |
| ------------ | ---------------------------- |
| Conv2D       | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2                          |
| Flatten      | Feature flattening           |
| Dense        | 100 units, ReLU              |
| Dense        | 10 units, Softmax            |

**Design choices**:

* Small kernel size for fine feature extraction
* ReLU activation for faster convergence
* Softmax for multi-class classification

---

## ⚙️ Training Configuration

* **Loss**: Categorical Cross-Entropy
* **Optimizer**: SGD (learning rate = 0.01, momentum = 0.9)
* **Batch size**: 64
* **Epochs**: 10
* **Validation split**: 10%

Training is fast and runs easily on CPU or Google Colab.

---

## 📈 Results

* **Test Accuracy**: ~99%
* Confusion matrix generated
* Misclassified samples analyzed

The model performs well on clean digits, with most errors occurring between visually similar digits (e.g., 4 vs 9).

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model

```bash
python train.py
```

The trained model will be saved to:

```
models/mnist_cnn.h5
```

### 3️⃣ Run inference

```bash
python inference.py --image path/to/digit.png
```

Example output:

```
Predicted digit : 7
Confidence      : 0.9923
```

---

## 📁 Project Structure

```
handwritten-digit-classifier/
├── train.py
├── inference.py
├── models/
│   └── mnist_cnn.h5
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🔮 Future Improvements

* Data augmentation
* CNN vs MLP comparison
* Grad-CAM visualizations
* Web demo using Hugging Face Spaces
* Model export to ONNX

---

## 👨‍💻 Author

Built as a portfolio project to demonstrate practical CNN and ML engineering skills.

---

⭐ If you find this project useful, consider starring the repository.