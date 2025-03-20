# BrainDx – Brain Tumour Classification 🧠

## Overview 📖
BrainDx is a machine learning-based project designed to classify brain tumors into different categories using medical imaging data. This tool leverages advanced deep learning techniques to assist in the early detection and diagnosis of brain tumors, thereby improving clinical decision-making.

## Features ✨
- **Automated Tumor Classification:** Classifies brain tumors into predefined categories (e.g., glioma, meningioma, pituitary tumor).
- **Deep Learning Models:** Utilizes state-of-the-art neural networks such as CNNs for image recognition.
- **User-Friendly Interface:** Provides an intuitive interface for uploading images and viewing predictions.
- **Evaluation Metrics:** Includes accuracy, precision, recall, and F1 score for model performance evaluation.

## Dataset 📊
The project uses a dataset of MRI brain scans with labeled tumor types. Ensure the dataset is preprocessed (e.g., resized, normalized) before training.

## Requirements 📋
Make sure you have the following dependencies installed:
- Python 3.8+
- TensorFlow or PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- OpenCV

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Installation 🚀

Cloning The Repo
```bash
git clone https://github.com/yourusername/BrainDx.git
```

Navigate to the project directory:
```bash
cd BrainDx
```

Install the Dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🏃‍♂️
Prepare your dataset and place it in the data/ directory.

Run the Jupyter notebook:
```bash
jupyter notebook notebooks/BrainDx_Notebook.ipynb
```

Example Command for Training:
```bash
python scripts/train_model.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

Example Command for Prediction:
```bash
python scripts/predict.py --image_path ./test_images/sample.jpg
```

## Results 📈
The model achieves an accuracy of 99.39% on the test set.


