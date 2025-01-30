# Pneumonia Detection with PyTorch 🩺

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project to detect pneumonia from chest X-ray images using PyTorch. This repository includes model training, evaluation, and a Flask-based web interface for real-time predictions.

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🚀 Project Overview
This project aims to automate pneumonia detection using a **CNN model** trained on [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). Key highlights:  
- **Accuracy**: 92% on test data.  
- **Tech Stack**: PyTorch (model training), OpenCV (data preprocessing), Flask (deployment).  
- **Impact**: Potential to reduce diagnostic delays in healthcare settings.  

## ✨ Features
- **Data Preprocessing**: Resizing, normalization, and augmentation of X-ray images.  
- **CNN Architecture**: Custom PyTorch model with convolutional layers for feature extraction.  
- **Flask API**: Real-time predictions via a user-friendly web interface.  
- **Error Handling**: Robust input validation and model performance optimization.  

## 🛠️ Installation
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/Asrar-ali/Pneumonia-Detection-PyTorch.git
   cd Pneumonia-Detection-PyTorch
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
3. **Download and organize the dataset:**:

   Download from Kaggle.
   Unzip and structure folders as:
   ```bash
   chest_xray/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      ├── test/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── val/
          ├── NORMAL/
          └── PNEUMONIA/
🖥️ Usage
1. Train the Model
      ```bash
      python train.py
  Model checkpoints saved as pneumonia_cnn.pth.
2. Evaluate Accuracy
      ```bash
      python evaluate.py
  Output:
3. Run the Flask Web App
   -Visit http://localhost:5000 to upload an X-ray image.
