# HandWrite AI — Handwritten Character Recognition

## Overview
HandWrite AI is an end-to-end machine learning web app that recognizes handwritten **digits (0–9)** and **letters (A–Z)** in real time. Draw on the canvas or upload an image; a Convolutional Neural Network (CNN) returns the predicted character and confidence scores.

This project completes the CodeAlpha Machine Learning task: *create a handwritten character recognition system that can recognize various handwritten characters or alphabets*.

## Problem Statement
Handwritten character recognition is a core computer-vision problem. MNIST covers digits only. This project combines **MNIST**, **EMNIST Letters**, and **EMNIST ByClass** (plus **USPS** when the public files download) so the same CNN can classify 36 classes (0–9 and A–Z), then serves the model through a Flask API and a browser UI.

## Features
- Interactive drawing canvas (mouse and touch)
- Image upload (`.png`, `.jpg`, `.jpeg`)
- Real-time predictions with top-3 probabilities
- Session history for the current browser session
- Glassmorphism UI and responsive layout
- Health endpoint to confirm the model is loaded

## Dataset
| Source | Content | Role |
| --- | --- | --- |
| MNIST | 60,000 train / 10,000 test digits | Classes `0`–`9` |
| EMNIST Letters | Handwritten A–Z | Classes `A`–`Z` (labels mapped after digits) |
| EMNIST ByClass | Digits, uppercase, and lowercase | Lowercase `a–z` mapped onto `A–Z`; extra digit styles |
| USPS | 16×16 digits resized to 28×28 | Optional extra digit source (downloaded when the public mirror is available) |

- Image size: 28×28 grayscale
- Classes: 36 (`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`)
- Default training uses a balanced subset (3,000 samples per class) plus shifted copies for off-center drawings

## Machine Learning Approach
CNN built with TensorFlow/Keras:

- On-the-fly BatchNorm CNN (32 / 64 / 128 filters)
- Dense 192 + Dropout 0.35
- Shifted training copies so off-center canvas strokes still match

Training uses Adam, `sparse_categorical_crossentropy`, EarlyStopping, and ReduceLROnPlateau.

## Data Preprocessing (web input)
1. Convert to grayscale
2. Invert if needed (model expects white strokes on black)
3. Crop to the character bounding box
4. Resize to 20×20 keeping aspect ratio
5. Center on a 28×28 canvas
6. Normalize pixels to `[0, 1]`

## Project Structure
```text
Character/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── install_deps.py
├── model/
│   ├── handwritten_character_cnn.keras
│   └── model_metadata.json
├── notebooks/
│   └── handwritten_digit_recognition.ipynb
├── training/
│   └── train_model.py
├── evaluation/
│   ├── evaluate_model.py
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_history.png
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
└── uploads/
```

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Fallback for flaky networks: `python install_deps.py`

3. Train the model (downloads MNIST, EMNIST splits, and USPS, writes `model/handwritten_character_cnn.keras`).
   A trained model is already included, so you can skip this step unless you want to retrain.
   The default run uses a balanced subset (3,000 samples per class) so training finishes on CPU:
   ```bash
   python training/train_model.py
   ```
   Full dataset (slower):
   ```bash
   python training/train_model.py --full
   ```

4. Evaluate:
   ```bash
   python evaluation/evaluate_model.py
   ```

5. Run the app:
   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000`

## Results
Evaluated on a held-out balanced **MNIST + EMNIST Letters** test set (500 samples per class, 18,000 images):

| Metric | Previous | Current |
| --- | --- | --- |
| Accuracy | 93.89% | **94.54%** |
| Precision | 93.95% | **94.60%** |
| Recall | 93.89% | **94.54%** |
| F1 Score | 93.87% | **94.51%** |

On a harder mixed test that also includes EMNIST ByClass (lowercase mapped to A–Z), accuracy is 93.06%. Digits stay near-perfect. The hardest letter pairs are still lookalikes (`I`/`L`, `G`/`Q`). Full per-class scores are in `evaluation/classification_report.txt`.

## Usage
1. **Draw**: sketch a digit or uppercase letter, then click Recognize.
2. **Upload**: drop a PNG/JPG of a single character, then click Analyze Image.
3. **History**: recent predictions appear on the right for this session.

## API
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Web UI |
| GET | `/health` | Model status and class list |
| POST | `/predict` | Canvas image (`image` file field) |
| POST | `/predict-upload` | Uploaded file (`file` field) |

## License
MIT License
