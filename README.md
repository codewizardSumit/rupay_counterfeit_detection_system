# 💵 Indian Currency: Real vs Fake Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
This project is an End-to-End Deep Learning Web Application designed to classify Indian Currency notes (ranging from ₹10 to ₹2000) as **Real** or **Fake**. Powered by a custom-built Convolutional Neural Network (CNN), the model extracts hidden features like security threads, watermarks, and micro-printing to make high-confidence predictions.

## ✨ Features
- **High Accuracy:** Custom CNN architecture trained to achieve ~93.8% validation accuracy.
- **Robust Feature Extraction:** Effectively ignores manual censorship/noise (e.g., black boxes) and focuses on core currency patterns using deep Convolutional layers and Dropout mechanisms.
- **Real-Time Web Interface:** A clean, user-friendly frontend built with Streamlit.
- **Confidence Scoring:** Outputs the prediction along with the AI's percentage of confidence.

## 🧠 Model Architecture & Strategy
The AI brain is built using **TensorFlow/Keras** with the following pipeline:
1. **Data Preprocessing:** Images are resized to `224x224` and pixel values are normalized `(0-1)` for faster convergence.
2. **Feature Extractors (CNN Blocks):** 4 blocks of `Conv2D + MaxPooling2D` to extract hierarchical features (from basic edges to complex RBI logos).
3. **The Classifier (Dense Layers):** A Fully Connected Artificial Neural Network with 1.3 Crore+ parameters.
4. **Anti-Overfitting:** Integrated `Dropout(0.5)` and `EarlyStopping` to ensure the model learns robust features rather than memorizing the training data.
5. **Loss & Optimizer:** `binary_crossentropy` and `Adam` optimizer (learning rate: 0.001).

## 🛠️ Tech Stack
- **Deep Learning Framework:** TensorFlow & Keras
- **Web Framework:** Streamlit
- **Data Manipulation:** NumPy, Pillow (PIL)
- **Deployment:** Render (Cloud PaaS)

## 📁 Project Structure
```text
├── data/                      # Dataset (Not pushed to GitHub due to size limits)
├── rupay_counterfeit_detector.keras  # Trained Model Weights (The AI Brain)
├── app.py                     # Streamlit Frontend Code
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation