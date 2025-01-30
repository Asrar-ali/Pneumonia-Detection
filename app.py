from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from model import PneumoniaCNN
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
model = PneumoniaCNN()
model.load_state_dict(torch.load('pneumonia_cnn.pth', map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    image = Image.open(image).convert('L')  # Convert to grayscale
    return transform(image).unsqueeze(0)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = preprocess_image(file)
    output = model(image)
    _, prediction = torch.max(output, 1)
    result = 'Pneumonia Detected' if prediction.item() == 1 else 'Normal'

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)