import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

TEST_DIR = "/mnt/c/Deepfake Image Detection/Final_Dataset/test"
MODEL_PATH = "/mnt/c/Deepfake Image Detection/saved_model/final_finetuned_model.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# LOAD TEST DATASET

def load_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    )

# TRUE LABELS

def get_y_true(test_ds):
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    return np.array(y_true)

# PREDICTIONS

def get_predictions(model, test_ds):
    preds = model.predict(test_ds).flatten()
    return preds

# PLOT PR CURVE

def plot_pr_curve(y_true, y_pred_probs, save_path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"Average Precision = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Loading test dataset...")
    test_ds = load_test_dataset()

    print("[INFO] Extracting true labels...")
    y_true = get_y_true(test_ds)

    print("[INFO] Predicting probabilities...")
    preds = get_predictions(model, test_ds)

    print("[INFO] Plotting PR curve...")
    plot_pr_curve(y_true, preds)

    print(" Saved PR curve as: pr_curve.png")

if __name__ == "__main__":
    main()
