import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

# PLOT ROC CURVE

def plot_roc_curve(y_true, y_pred_probs, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
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

    print("[INFO] Plotting ROC curve...")
    plot_roc_curve(y_true, preds)

    print(" Saved ROC curve as: roc_curve.png")

if __name__ == "__main__":
    main()

