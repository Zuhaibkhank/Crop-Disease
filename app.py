import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import gdown

MODEL_PATH = "model/crop_model.keras"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1Pk--I6eXVErjViq0uGO_BoiE_zRNxeqW"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

print("Loading model...")
try:
    model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded with tf_keras")
except Exception as e:
    print(f"tf_keras failed: {e}, trying tf.keras...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded with tf.keras")

# Load classes
with open("model/classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"✅ {len(class_names)} classes loaded")

app = Flask(__name__)

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return class_names[index], confidence

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')

    if not file or file.filename == '':
        return "No file selected", 400

    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    clean_name = result.replace("___", " - ").replace("_", " ")
    info = f"Detected: {clean_name}. Please take proper treatment."

    return render_template(
        "result.html",
        prediction=result,
        clean_name=clean_name,
        confidence=round(confidence, 1),
        info=info,
        img_path=filepath
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)