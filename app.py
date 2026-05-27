from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import zipfile
import os

from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# =========================
# DOWNLOAD MODEL DARI DRIVE
# =========================

if not os.path.exists("saved_model"):

    file_id = "14mBvkmgggcTQw3vYwWRF54oHddVIORxn"

    url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading model...")

    gdown.download(url, "saved_model.zip", quiet=False)

    print("Extracting model...")

    with zipfile.ZipFile("saved_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# =========================
# LOAD MODEL
# =========================

model = tf.saved_model.load("saved_model/saved_model")

infer = model.signatures["serving_default"]

# =========================
# CLASS LABEL
# =========================

class_names = [
    "Abyssinian",
    "American_Curl",
    "American_Shorthair",
    "Bengal",
    "Birman",
    "British_Shorthair",
    "Egyptian_Mau",
    "Exotic_Shorthair",
    "Maine_Coon",
    "Manx",
    "Munchkin",
    "Norwegian_Forest",
    "Persia",
    "Ragdoll",
    "Russian_Blue",
    "Scottish_Fold",
    "Siamese",
    "Sphynx",
    "Toyger",
    "Turkish_Angora"
]

# =========================
# HOME ROUTE
# =========================

@app.route("/")
def home():
    return "Model Berhasil Dimuat!"

# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Ambil file gambar
        file = request.files["file"]

        # Open image
        img = Image.open(file).convert("RGB")

        # Resize
        img = img.resize((224, 224))

        # Convert ke numpy
        img_array = np.array(img)

        # Preprocessing ResNet50
        img_array = preprocess_input(img_array)

        # Tambah batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = infer(tf.constant(img_array))

        # Ambil output tensor
        output = list(prediction.values())[0].numpy()

        # Ambil index kelas tertinggi
        predicted_class = np.argmax(output)

        # Confidence
        confidence = float(np.max(output) * 100)

        # Result JSON
        result = {
            "class": class_names[predicted_class],
            "confidence": round(confidence, 2)
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================
# RUN FLASK
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)