import argparse
import gzip
import json
import os
import struct

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import get_file

CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASS_NAMES)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

DATASET_INFO = {
    "train_sources": {},
    "test_sources": {},
}

EMNIST_MIRRORS = [
    "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/",
    "https://huggingface.co/datasets/Royc30ne/emnist-letters/resolve/main/",
]

USPS_TRAIN_URLS = [
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz",
    "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz",
    "https://www-stat-class.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz",
    "https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz",
]
USPS_TEST_URLS = [
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz",
    "https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz",
    "https://www-stat-class.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz",
    "https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz",
]

# EMNIST Balanced extra classes are lowercase letters not merged with uppercase.
BALANCED_EXTRA_TO_36 = {
    36: 10,  # a -> A
    37: 11,  # b -> B
    38: 13,  # d -> D
    39: 14,  # e -> E
    40: 15,  # f -> F
    41: 16,  # g -> G
    42: 17,  # h -> H
    43: 23,  # n -> N
    44: 26,  # q -> Q
    45: 27,  # r -> R
    46: 29,  # t -> T
}


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic: {magic} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


def _peek_idx_magic(path):
    with gzip.open(path, "rb") as f:
        return struct.unpack(">I", f.read(4))[0]


def _download_emnist(filename, cache_subdir):
    last_error = None
    for base in EMNIST_MIRRORS:
        try:
            path = get_file(
                fname=filename,
                origin=base + filename,
                cache_subdir=cache_subdir,
                file_hash=None,
            )
            magic = _peek_idx_magic(path)
            if magic in (2049, 2051):
                return path
            print(f"Downloaded file was not a valid IDX archive (magic={magic}). Trying next mirror...")
            os.remove(path)
        except Exception as exc:
            last_error = exc
            print(f"Failed to download {filename} from {base}: {exc}")
    raise RuntimeError(f"Could not download {filename}. Last error: {last_error}")


def _download_url(filename, urls, cache_subdir):
    last_error = None
    for origin in urls:
        try:
            return get_file(
                fname=filename,
                origin=origin,
                cache_subdir=cache_subdir,
                file_hash=None,
            )
        except Exception as exc:
            last_error = exc
            print(f"Failed to download {filename} from {origin}: {exc}")
    raise RuntimeError(f"Could not download {filename}. Last error: {last_error}")


def _rotate_emnist(images):
    return np.transpose(images, (0, 2, 1))


def _resize_to_28(images):
    out = np.empty((images.shape[0], 28, 28), dtype=np.uint8)
    for i, image in enumerate(images):
        out[i] = np.array(Image.fromarray(image).resize((28, 28), Image.Resampling.LANCZOS))
    return out


def _map_balanced_labels(labels):
    mapped = labels.astype(np.int32).copy()
    for source, target in BALANCED_EXTRA_TO_36.items():
        mapped[labels == source] = target
    return mapped


def _map_byclass_labels(labels):
    mapped = labels.astype(np.int32).copy()
    lowercase = mapped >= 36
    mapped[lowercase] = mapped[lowercase] - 26
    return mapped


def _balanced_subset(x, y, max_per_class, seed=42):
    if max_per_class is None:
        return x, y

    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        class_idx = np.where(y == class_id)[0]
        if class_idx.size == 0:
            raise ValueError(f"No samples found for class {class_id} ({CLASS_NAMES[class_id]})")
        if class_idx.size > max_per_class:
            class_idx = rng.choice(class_idx, max_per_class, replace=False)
        selected.append(class_idx)

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return x[indices], y[indices]


def _concat_sources(parts):
    names, images, labels = zip(*parts)
    x = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    counts = {name: int(arr.shape[0]) for name, arr in zip(names, images)}
    return x, y, counts


def _load_mnist(include_train=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if not include_train:
        return None, None, x_test, y_test.astype(np.int32)
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)


def _load_emnist_split(prefix, cache_subdir, label_mapper, include_train=True):
    test_images = _download_emnist(f"{prefix}-test-images-idx3-ubyte.gz", cache_subdir)
    test_labels = _download_emnist(f"{prefix}-test-labels-idx1-ubyte.gz", cache_subdir)
    x_test = _rotate_emnist(_read_idx_images(test_images))
    y_test = label_mapper(_read_idx_labels(test_labels))

    if not include_train:
        return None, None, x_test, y_test

    train_images = _download_emnist(f"{prefix}-train-images-idx3-ubyte.gz", cache_subdir)
    train_labels = _download_emnist(f"{prefix}-train-labels-idx1-ubyte.gz", cache_subdir)
    x_train = _rotate_emnist(_read_idx_images(train_images))
    y_train = label_mapper(_read_idx_labels(train_labels))
    return x_train, y_train, x_test, y_test


def _load_usps(include_train=True):
    test_path = _download_url("zip.test.gz", USPS_TEST_URLS, os.path.join("datasets", "usps"))

    def read_split(path):
        with gzip.open(path, "rt") as handle:
            data = np.loadtxt(handle)
        labels = data[:, 0].astype(np.int32)
        images = data[:, 1:].reshape(-1, 16, 16)
        images = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)
        images = _resize_to_28(images)
        return images, labels

    x_test, y_test = read_split(test_path)
    if not include_train:
        return None, None, x_test, y_test

    train_path = _download_url("zip.train.gz", USPS_TRAIN_URLS, os.path.join("datasets", "usps"))
    x_train, y_train = read_split(train_path)
    return x_train, y_train, x_test, y_test


def load_character_dataset(max_per_class=None, include_test_cap=None, include_train=True, include_extra=True):
    """Load and merge handwriting datasets into 36 classes (0-9, A-Z)."""
    import gc

    train_parts = []
    test_parts = []

    print("Loading MNIST digits...")
    x_mnist_train, y_mnist_train, x_mnist_test, y_mnist_test = _load_mnist(include_train)
    if include_train:
        train_parts.append(("MNIST", x_mnist_train, y_mnist_train))
    test_parts.append(("MNIST", x_mnist_test, y_mnist_test))

    print("Loading EMNIST letters (A-Z)...")
    x_let_train, y_let_train, x_let_test, y_let_test = _load_emnist_split(
        "emnist-letters",
        os.path.join("datasets", "emnist-letters"),
        lambda labels: labels.astype(np.int32) - 1 + 10,
        include_train=include_train,
    )
    if include_train:
        train_parts.append(("EMNIST Letters", x_let_train, y_let_train))
    test_parts.append(("EMNIST Letters", x_let_test, y_let_test))

    if include_extra:
        try:
            print("Loading EMNIST ByClass (digits + A-Z + a-z mapped to A-Z)...")
            x_bc_train, y_bc_train, x_bc_test, y_bc_test = _load_emnist_split(
                "emnist-byclass",
                os.path.join("datasets", "emnist-byclass"),
                _map_byclass_labels,
                include_train=include_train,
            )
            if include_train:
                x_bc_train, y_bc_train = _balanced_subset(x_bc_train, y_bc_train, 2500, seed=11)
                train_parts.append(("EMNIST ByClass", x_bc_train, y_bc_train))
            x_bc_test, y_bc_test = _balanced_subset(x_bc_test, y_bc_test, 300, seed=13)
            test_parts.append(("EMNIST ByClass", x_bc_test, y_bc_test))
        except Exception as exc:
            print(f"Skipping EMNIST ByClass: {exc}")

        try:
            print("Loading USPS digits...")
            x_usps_train, y_usps_train, x_usps_test, y_usps_test = _load_usps(include_train)
            if include_train:
                train_parts.append(("USPS", x_usps_train, y_usps_train))
            test_parts.append(("USPS", x_usps_test, y_usps_test))
        except Exception as exc:
            print(f"Skipping USPS: {exc}")

    x_test, y_test, test_counts = _concat_sources(test_parts)
    x_test, y_test = _balanced_subset(x_test, y_test, include_test_cap, seed=7)
    DATASET_INFO["test_sources"] = test_counts

    if not include_train:
        rng = np.random.default_rng(42)
        test_idx = rng.permutation(len(x_test))
        DATASET_INFO["train_sources"] = {}
        empty_x = np.empty((0, 28, 28), dtype=np.uint8)
        empty_y = np.empty((0,), dtype=np.int32)
        gc.collect()
        return empty_x, empty_y, x_test[test_idx], y_test[test_idx]

    x_train, y_train, train_counts = _concat_sources(train_parts)
    del train_parts
    x_train, y_train = _balanced_subset(x_train, y_train, max_per_class, seed=42)
    DATASET_INFO["train_sources"] = train_counts
    gc.collect()

    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(x_train))
    test_idx = rng.permutation(len(x_test))
    return x_train[train_idx], y_train[train_idx], x_test[test_idx], y_test[test_idx]


def _shift_augment(x, y, fraction=0.3, seed=0):
    """Add translated copies so the model is more robust to off-center drawings."""
    rng = np.random.default_rng(seed)
    n = max(1, int(len(x) * fraction))
    idx = rng.choice(len(x), size=n, replace=False)
    extras = np.empty((n,) + x.shape[1:], dtype=x.dtype)
    for i, source in enumerate(idx):
        dy, dx = rng.integers(-2, 3, size=2)
        extras[i] = np.roll(np.roll(x[source], int(dy), axis=0), int(dx), axis=1)
    return np.concatenate([x, extras], axis=0), np.concatenate([y, y[idx]], axis=0)


def build_model():
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2),
            Conv2D(64, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2),
            Conv2D(128, 3, activation="relu"),
            BatchNormalization(),
            Flatten(),
            Dense(192, activation="relu"),
            Dropout(0.35),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _save_history_plot(history, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping training history plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Training history saved to {out_path}")


def train(full=False):
    import gc

    max_per_class = None if full else 3000
    test_cap = None if full else 500
    epochs = 12 if full else 8
    batch_size = 128 if full else 256

    x_train, y_train, x_test, y_test = load_character_dataset(
        max_per_class=max_per_class,
        include_test_cap=test_cap,
        include_extra=True,
    )

    print(f"Training sources: {DATASET_INFO['train_sources']}")
    print(f"Testing sources: {DATASET_INFO['test_sources']}")
    print(f"Training shape: {x_train.shape}")
    print(f"Testing shape: {x_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train, minlength=NUM_CLASSES)}")

    print("Preprocessing data...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print("Adding shifted copies for off-center robustness...")
    x_train, y_train = _shift_augment(x_train, y_train, fraction=0.3, seed=3)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    gc.collect()
    print(f"Training shape after augmentation: {x_train.shape}")

    print("Building model...")
    model = build_model()
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    candidate_path = os.path.join(MODEL_DIR, "handwritten_character_cnn.candidate.keras")

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-5)
    checkpoint = ModelCheckpoint(candidate_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr, checkpoint],
    )

    print("Evaluating on mixed test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Mixed Test Accuracy: {test_acc:.4f}")
    train_sources = dict(DATASET_INFO.get("train_sources") or {})
    mixed_test_sources = dict(DATASET_INFO.get("test_sources") or {})

    print("Evaluating on core MNIST + EMNIST letters test set...")
    _, _, x_core, y_core = load_character_dataset(
        include_train=False,
        include_test_cap=500,
        include_extra=False,
    )
    x_core = np.expand_dims(x_core.astype("float32") / 255.0, -1)
    core_loss, core_acc = model.evaluate(x_core, y_core, verbose=0)
    print(f"Core Test Accuracy: {core_acc:.4f}")

    model.save(candidate_path)
    print(f"Candidate model saved to {candidate_path}")
    _save_history_plot(history, os.path.join(EVAL_DIR, "training_history.png"))

    model_path = os.path.join(MODEL_DIR, "handwritten_character_cnn.keras")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    previous_acc = 0.0
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                previous_acc = float(json.load(f).get("test_accuracy", 0.0) or 0.0)
        except Exception:
            previous_acc = 0.0

    promoted = core_acc >= previous_acc
    if promoted:
        model.save(model_path)
        print(f"Promoted new model ({core_acc:.4f} >= previous {previous_acc:.4f}) to {model_path}")
    else:
        print(
            f"Keeping previous model ({previous_acc:.4f}); candidate core accuracy was {core_acc:.4f}."
        )

    metadata = {
        "dataset": "MNIST + EMNIST Letters + EMNIST ByClass + USPS",
        "train_sources": train_sources,
        "test_sources": mixed_test_sources,
        "class_names": CLASS_NAMES,
        "training_samples": int(x_train.shape[0]),
        "testing_samples": int(x_test.shape[0]),
        "balanced_subset": not full,
        "max_per_class": max_per_class,
        "image_size": [28, 28],
        "num_classes": NUM_CLASSES,
        "epochs_trained": len(history.history["loss"]),
        "augmentation": ["pixel_shift"],
        "mixed_test_accuracy": round(float(test_acc), 4),
        "mixed_test_loss": round(float(test_loss), 4),
        "core_test_accuracy": round(float(core_acc), 4),
        "core_test_loss": round(float(core_loss), 4),
        "test_accuracy": round(float(core_acc if promoted else previous_acc), 4),
        "test_loss": round(float(core_loss), 4),
        "promoted": bool(promoted),
    }

    if promoted:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")
    else:
        candidate_meta = os.path.join(MODEL_DIR, "candidate_metadata.json")
        with open(candidate_meta, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Candidate metrics saved to {candidate_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the handwritten character CNN.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the complete combined dataset with no per-class cap (slower, especially on CPU).",
    )
    args = parser.parse_args()
    train(full=args.full)
