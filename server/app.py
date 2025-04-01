import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Get absolute path for models directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models/")

# Load all models
MODEL_MAPPING = {
    "apple": "model_apple.h5",
    "bell_pepper": "model_bell_pepper.h5",
    "cherry": "model_cherry.h5",
    "corn": "model_corn.h5",
    "grape": "model_grape.h5",
    "potato": "model_potato.h5",
    "tomato": "model_tomato.h5",
}

models = {}
for plant, model_name in MODEL_MAPPING.items():
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        models[plant] = tf.keras.models.load_model(model_path)
    else:
        print(f"⚠️ Model not found: {model_path}")

# Load disease information
disease_info_path = os.path.join(MODELS_DIR, "disease_info.json")
if os.path.exists(disease_info_path):
    with open(disease_info_path, "r") as f:
        disease_info = json.load(f)
else:
    print(f"⚠️ Disease info JSON not found: {disease_info_path}")

# Image preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def home():
    return "Plant Disease Detector API is running!"

# API Route: Upload Image and Predict Disease
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files or "plant" not in request.form:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        plant_species = request.form["plant"].lower()

        if plant_species not in models:
            return jsonify({"error": f"No model found for {plant_species}"}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        img_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(img_path)

        # Preprocess image and predict
        img_array = preprocess_image(img_path)
        model = models[plant_species]
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Get disease and treatment
        disease = list(disease_info[plant_species].keys())[predicted_class]
        treatment = disease_info[plant_species][disease]

        return jsonify({
            "plant": plant_species,
            "disease": disease,
            "treatment": treatment,
        })
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")  # Log error to console
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
