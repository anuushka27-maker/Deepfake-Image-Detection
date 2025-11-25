import tensorflow as tf

MODEL_PATH = "/mnt/c/Deepfake Image Detection/saved_model/final_finetuned_model.h5"

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

if __name__ == "__main__":
    load_model()