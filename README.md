
# 🌱 Plant Leaf Disease Image Classification

A Deep Learning and Computer Vision project focused on classifying plant leaf diseases from images using Convolutional Neural Networks (CNN) and MobileNetV2 transfer learning.

## 🎯 Project Overview

Plant diseases can affect agricultural productivity and crop quality. This project aims to build an image classification system that identifies healthy and diseased plant leaves using deep learning techniques.

The project explores image preprocessing, CNN model development, transfer learning, and model evaluation.

## 🚀 Objectives

- Explore and prepare a labelled plant leaf image dataset.
- Perform image preprocessing and augmentation.
- Build a custom Convolutional Neural Network (CNN).
- Implement MobileNetV2 using transfer learning.
- Compare model performance using evaluation metrics.
- Develop an interactive application for leaf disease classification.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Development environment |
| TensorFlow | Deep learning framework |
| Keras | Model development API |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Interactive application (planned) |

## 📊 Dataset

This project uses the **PlantVillage Dataset**, a publicly available collection of labelled plant leaf images.

**Dataset:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

The initial classification experiment will focus on five tomato leaf categories:

1. Tomato Healthy
2. Tomato Bacterial Spot
3. Tomato Early Blight
4. Tomato Late Blight
5. Tomato Leaf Mold

The final image counts will be determined during dataset exploration and preprocessing.

## 🧠 Models

### 1. Custom CNN
A Convolutional Neural Network developed and trained from scratch to establish a baseline classification model.

### 2. MobileNetV2
A pretrained deep learning model using transfer learning to extract image features and classify plant leaf categories.

Both approaches will be evaluated and compared.

## 📈 Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Correct and Incorrect Prediction Analysis

Evaluation results will be added after model training.

## 📂 Project Structure

```text
Plant-Leaf-Disease-Classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_custom_cnn.ipynb
│   ├── 04_mobilenetv2.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/
├── reports/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Plant-Leaf-Disease-Classification.git
```

### 2. Navigate to the project directory

```bash
cd Plant-Leaf-Disease-Classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 🖥️ Interactive Demo

An interactive application is planned where users will be able to upload a plant leaf image and receive a predicted classification from the trained model.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions should not be considered professional agricultural advice or a substitute for expert plant disease diagnosis.

## 📌 Project Status

**Currently in Development 🚧**

Dataset preparation, model development, evaluation, and application implementation are being carried out progressively.

## 👨‍💻 Author

**Abhay Yadav**

- GitHub: [i-abhayy](https://github.com/i-abhayy)
- LinkedIn: [Abhay Yadav](https://www.linkedin.com/in/i-abhay/)

---

⭐ If you find this project interesting, consider giving the repository a star!
