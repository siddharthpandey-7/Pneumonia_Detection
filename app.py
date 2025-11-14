import os
import io
import base64
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# ------------------ MODEL DOWNLOAD CONFIG ------------------
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.keras"
MODEL_PATH = "best_vgg19_pneumonia.keras"


def download_model():
    """Download model from HuggingFace if not present."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000000:
        print("Downloading model from HuggingFace...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception("Failed to download model.")


model = None


def get_model():
    """Loads the model only once."""
    global model
    if model is None:
        download_model()
        print("Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        # Preprocess image
        image = Image.open(file).convert("RGB")
        image_resized = image.resize((128, 128))
        arr = np.array(image_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        model_instance = get_model()
        preds = model_instance.predict(arr)

        # Softmax output shape → [Normal, Pneumonia]
        if preds.shape[-1] == 2:
            pneumonia_prob = float(preds[0][1])
        else:
            pneumonia_prob = float(preds[0][0])

        result = "PNEUMONIA DETECTED" if pneumonia_prob > 0.5 else "NORMAL"
        confidence = round(
            pneumonia_prob * 100 if pneumonia_prob > 0.5 else (1 - pneumonia_prob) * 100, 2
        )

        # Convert uploaded image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_data = f"data:image/png;base64,{img_encoded}"

        return render_template(
            "result.html",
            prediction=result,
            confidence=confidence,
            img_data=img_data
        )

    except Exception as e:
        print("Prediction error:", e)
        return f"Error during prediction: {str(e)}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
