import os
import json
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("model", exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "handwritten_character_cnn.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
model = None


def load_class_names():
    global CLASS_NAMES
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            if metadata.get("class_names"):
                CLASS_NAMES = metadata["class_names"]
        except Exception as e:
            print(f"Could not read metadata: {e}")


def load_model_once():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")


load_class_names()
load_model_once()


def preprocess_image(image, is_canvas=False):
    img = image.convert("L")

    if is_canvas:
        img = ImageOps.invert(img)
    else:
        stat = np.array(img)
        if np.mean(stat[0, :]) > 127 or np.mean(stat[-1, :]) > 127:
            img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (28, 28), color=0)
    paste_pos = ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2)
    new_img.paste(img, paste_pos)

    img_array = np.array(new_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def predict_from_array(processed_img):
    predictions = model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        {
            "index": int(i),
            "character": CLASS_NAMES[int(i)],
            "digit": CLASS_NAMES[int(i)],
            "confidence": round(float(predictions[i] * 100), 2),
        }
        for i in top_indices
    ]
    best = top_indices[0]
    return {
        "prediction": CLASS_NAMES[int(best)],
        "character": CLASS_NAMES[int(best)],
        "confidence": round(float(predictions[best] * 100), 2),
        "top_predictions": top_predictions,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    })


@app.route("/predict", methods=["POST"])
def predict_canvas():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty image"}), 400

    try:
        image = Image.open(file.stream)
        processed_img = preprocess_image(image, is_canvas=True)
        return jsonify(predict_from_array(processed_img))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_exts = {"png", "jpg", "jpeg"}
    if not ("." in file.filename and file.filename.rsplit(".", 1)[1].lower() in allowed_exts):
        return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG"}), 400

    try:
        image = Image.open(file.stream)
        processed_img = preprocess_image(image, is_canvas=False)
        return jsonify(predict_from_array(processed_img))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
