import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

TEST_DIR = "/mnt/c/Deepfake Image Detection/Final_Dataset/test"    
MODEL_PATH = "/mnt/c/Deepfake Image Detection/saved_model/final_finetuned_model.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 20

# LOAD TEST DATASET

def load_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    )

# GET TRUE LABELS

def get_y_true(ds):
    labels = []
    for _, y in ds:
        labels.extend(y.numpy())
    return np.array(labels)

# GET PREDICTIONS

def get_predictions(model, ds):
    prob = model.predict(ds).flatten()       # continuous output
    classes = (prob >= 0.5).astype(int)      # convert → 0 or 1
    return prob, classes

# PLOT CONFUSION MATRIX

def plot_confusion_matrix(y_true, y_pred_classes, save_path="conf_matrix.png"):
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                cmap="Blues", xticklabels=["Real","Fake"],
                yticklabels=["Real","Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Loading Test Data...")
    test_ds = load_test_dataset()

    print("[INFO] Extracting True Labels...")
    y_true = get_y_true(test_ds)

    print("[INFO] Predicting...")
    y_prob, y_pred_classes = get_predictions(model, test_ds)

    print("[INFO] Plotting Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred_classes)

    print(" Confusion matrix saved as conf_matrix.png")

if __name__ == "__main__":
    main()
