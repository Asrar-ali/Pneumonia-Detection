# Container image for the Pneumonia Detection Flask API
FROM python:3.10-slim

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

EXPOSE 5000

# Serve the Flask app with Gunicorn (expects pneumonia_cnn.pth to be present,
# e.g. baked into the image or mounted at runtime)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
