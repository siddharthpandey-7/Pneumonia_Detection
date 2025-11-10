import os
import requests
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"

# ------------------ DOWNLOAD MODEL --------------------
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        print("✅ Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)

        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception("❌ ERROR: Failed to download model from Hugging Face.")

# ------------------ LOAD MODEL --------------------
download_model()

try:
    model = load_model(MODEL_PATH)
    print("✅ Loaded model successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e

# ------------------ ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    result = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
