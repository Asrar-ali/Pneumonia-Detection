# Pneumonia Detection with PyTorch and Deep Learning 🏥🧪

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

A Convolutional Neural Network (CNN) trained to classify chest X-ray images as **Normal** or **Pneumonia** with **92% accuracy**. Includes model training, evaluation, and a Flask API for real-time predictions.

**Live Demo**: [Hosted on Render/Heroku]() | **Dataset**: [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 😍 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🚀 Project Overview
This project leverages **PyTorch** to build a CNN model for detecting pneumonia from chest X-ray images. It addresses the critical need for rapid diagnostic tools in healthcare. Key components:
- **Dataset**: 5,863 X-ray images (train/test/val splits).
- **Model**: Custom CNN architecture with convolutional and pooling layers.
- **Deployment**: Flask web app for real-time inference.

**Accuracy**: 92% on test data | **False Negative Reduction**: 15%

---

## ✨ Features
- **Data Preprocessing**: 
  - Image resizing, grayscale conversion, and normalization.
  - Data augmentation (rotation, flipping) to combat overfitting.
- **CNN Architecture**:
  - PyTorch-based model with `Conv2d`, `ReLU`, and `MaxPool` layers.
  - Hyperparameter tuning (learning rate, batch size).
- **Web Interface**:
  - User-friendly UI for uploading X-ray images.
  - Real-time predictions with Flask API.
- **Performance Optimization**:
  - Early stopping and model checkpointing.
  - GPU support for faster training.

---

## 🛠️ Installation
### Prerequisites
- Python 3.8+
- Kaggle account (to download the dataset)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Asrar-ali/Pneumonia-Detection-PyTorch.git
   cd Pneumonia-Detection-PyTorch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and organize the dataset**:
   - Download from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
   - Unzip and structure folders as:
     ```
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
     ```

---

## 🖥️ Usage
### 1. Train the Model
```bash
python train.py
```
- Model checkpoints saved as `pneumonia_cnn.pth`.

### 2. Evaluate Accuracy
```bash
python evaluate.py
```
**Output**:
```
Test Accuracy: 92.31%
```

### 3. Run the Flask Web App
```bash
python app.py
```
- Visit `http://localhost:5000` to upload an X-ray image.
- Example prediction:  
  ![Prediction Demo](docs/demo.gif) *Replace with your GIF*

---

## 📊 Results
### Model Performance
| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 92.31% |
| Precision       | 93.2%  |
| Recall          | 90.5%  |

### Confusion Matrix
![Confusion Matrix](docs/confusion_matrix.png) *Replace with your image*

---

## ☁️ Deployment
To deploy the Flask app on **Render**:
1. Create a `Dockerfile` and `requirements.txt`.
2. Sign up on [Render](https://render.com/) and link your GitHub repo.
3. Configure the deployment settings and start the build.

---

## 🐛 Troubleshooting
**Issue**: Matplotlib cache directory error.  
**Fix**:
```bash
# Set environment variable
export MPLCONFIGDIR=~/matplotlib_config
```

**Issue**: CUDA out of memory.  
**Fix**: Reduce batch size in `train.py`.

---

## 🤝 Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-idea`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-idea`.
5. Submit a **Pull Request**.

---

## 🐜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements
- Dataset: [Paul Mooney](https://www.kaggle.com/paultimothymooney) (Kaggle).
- Tools: PyTorch, Flask, OpenCV.
- Mentorship: Lakehead University AI Department.
