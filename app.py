import os
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Hugging Face model URL (you uploaded this)
MODEL_URL = "https://huggingface.co/siddharthpandey7/pneumonia-model/resolve/main/best_vgg19_pneumonia.h5"
MODEL_PATH = "best_vgg19_pneumonia.h5"


# --- Download model if not found locally ---
def download_model():
    print("🧠 Checking for model...")
    if not os.path.exists(MODEL_PATH):
        print("⏬ Model not found locally. Downloading from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise Exception(f"❌ Model download failed! HTTP {response.status_code}")
    else:
        print("✅ Model already exists locally.")


# --- Load model safely ---
def load_trained_model():
    download_model()
    print("🔄 Loading model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return model


# Load model once at startup
model = load_trained_model()

# --- Class names ---
class_names = ['NORMAL', 'PNEUMONIA']


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    return render_template('result.html',
                           filename=file.filename,
                           prediction=predicted_class,
                           confidence=confidence)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
