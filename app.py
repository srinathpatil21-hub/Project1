import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# --- Configuration (Must match the training script) ---
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'plant_disease_model.h5'
LABELS_PATH = 'class_labels.json'

# --- 1. Initialize Flask App and Load Model ---
app = Flask(__name__)
# Enable CORS for the frontend to communicate with the backend
CORS(app) 

# Global variables for model and labels
model = None
class_labels = {}

def load_resources():
    """Load the Keras model and class labels into memory."""
    global model, class_labels
    try:
        # Load the model
        model = load_model(MODEL_PATH, compile=False)
        # Load the class labels (disease names)
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print("Model and labels loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None
        class_labels = {}

def preprocess_image(image_bytes):
    """
    Preprocesses the image bytes to be ready for the CNN model.
    """
    # Open the image from bytes
    image = Image.open(BytesIO(image_bytes))
    
    # 1. Convert to RGB (if not already)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # 2. Resize to the model's expected input size
    image = image.resize(IMAGE_SIZE)
    
    # 3. Convert to a NumPy array and normalize
    # Model was trained with normalization (1./255)
    img_array = np.array(image) / 255.0
    
    # 4. Reshape to (1, 128, 128, 3) for the model
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route("/", methods=["GET"])
def index():
    """Simple health check endpoint."""
    if model:
        return jsonify({"status": "Model loaded", "message": "API is ready for predictions."})
    else:
        return jsonify({"status": "Error", "message": "Model failed to load."}), 500

# --- 2. Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Get image bytes
            image_bytes = file.read()
            
            # Preprocess the image
            processed_image = preprocess_image(image_bytes)
            
            # Make the prediction
            predictions = model.predict(processed_image)
            
            # Get the class index with the highest probability
            predicted_class_index = np.argmax(predictions[0])
            
            # Convert index to class label (disease name)
            # class_labels is stored as {index: label_name}
            disease_name = class_labels.get(str(predicted_class_index), "Unknown Disease")
            
            # Get the confidence score
            confidence = float(predictions[0][predicted_class_index])
            
            # Return the result as JSON
            return jsonify({
                "prediction": disease_name,
                "confidence": f"{confidence * 100:.2f}%"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Load the model and labels when the Flask app starts
    load_resources() 
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
