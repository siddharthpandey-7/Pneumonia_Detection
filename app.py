import os
import requests
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Hugging Face model link
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"

# ------------------ DOWNLOAD MODEL ------------------
def download_model():
    """Download the model if not already present."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        print("🧠 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception("❌ ERROR: Failed to download model from Hugging Face.")

# ------------------ LOAD MODEL (Lazy Loading) ------------------
model = None

def get_model():
    """Load the model only once (lazy load)."""
    global model
    if model is None:
        print("🧩 Loading model into memory...")
        download_model()
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
    return model

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        # ✅ Save uploaded file for display
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # ✅ Preprocess image (match model input)
        img = Image.open(filepath).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_instance = get_model()

        print("🧠 Predicting...")
        prediction = model_instance.predict(img_array)

        # ✅ Handle model output
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            confidence = float(prediction[0][0])
            result = "PNEUMONIA" if confidence > 0.5 else "NORMAL"
            confidence_percent = round(confidence * 100 if confidence > 0.5 else (1 - confidence) * 100, 2)
        elif prediction.ndim == 2 and prediction.shape[1] == 2:
            predicted_class = np.argmax(prediction)
            confidence_percent = round(np.max(prediction) * 100, 2)
            result = "PNEUMONIA" if predicted_class == 1 else "NORMAL"
        else:
            result = "UNKNOWN"
            confidence_percent = 0.0

        print(f"✅ Prediction complete: {result} ({confidence_percent}%)")

        return render_template(
            "result.html",
            filename=file.filename,
            prediction=result,
            confidence=confidence_percent
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        import traceback
        traceback.print_exc()
        return f"Error during prediction: {str(e)}", 500

# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
