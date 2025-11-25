import tensorflow as tf
import numpy as np
import os
from keras.preprocessing import image

# 1. MODEL PATH

MODEL_PATH = "/mnt/c/Deepfake Image Detection/saved_model/final_finetuned_model.h5"

# 2. LOAD MODEL

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# 3. PREDICT FUNCTION

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]

    label = "Fake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred

    print("\n==============================")
    print(f" Prediction: {label}")
    print(f" Confidence: {confidence * 100:.2f}%")
    print("==============================\n")

# 4. MAIN SCRIPT

if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()

    if not os.path.exists(img_path):
        print(" ERROR: File not found:", img_path)
    else:
        predict_image(img_path)
