from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
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
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model Loaded")

# Load classes
with open("model/classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

app = Flask(__name__)

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    return class_names[index]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file.filename == '':
        return "No file selected"

    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    clean_name = result.replace("___", " - ").replace("_", " ")
    info = f"Detected: {clean_name}. Please take proper treatment."

    return render_template(
        "result.html",
        prediction=result,
        info=info,
        img_path=filepath
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)