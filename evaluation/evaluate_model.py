import os
import json
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from train_model import CLASS_NAMES, load_character_dataset


def evaluate():
    _, _, x_test, y_test = load_character_dataset(
        include_train=False,
        include_test_cap=500,
        include_extra=False,
    )
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "model", "handwritten_character_cnn.keras")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Running predictions...")
    y_pred_probs = model.predict(x_test, batch_size=256, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Calculating metrics...")
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n--- Model Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("---------------------\n")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print("Classification Report:")
    print(report)

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(eval_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Model Metrics ---\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write("---------------------\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix - Handwritten Characters")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    metadata_path = os.path.join(project_root, "model", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata["precision"] = round(float(precision), 4)
        metadata["recall"] = round(float(recall), 4)
        metadata["f1_score"] = round(float(f1), 4)
        metadata["test_accuracy"] = round(float(acc), 4)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Updated {metadata_path} with precision, recall, and f1_score.")


if __name__ == "__main__":
    evaluate()
