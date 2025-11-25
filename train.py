# train.py 

import os, sys, random, json, gc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

# 0️ CLEAN GPU VRAM BEFORE TRAINING

print(" Cleaning GPU VRAM and resetting session...")
tf.keras.backend.clear_session()
gc.collect()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(" GPU memory growth enabled")
else:
    print(" No GPU detected!")

# 1️ SEED FIX

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# 2️ DISABLE MIXED PRECISION (Guided Backprop Safe)

from keras import mixed_precision
mixed_precision.set_global_policy("float32")

# 3️ LOAD YOUR MODEL ARCHITECTURE

sys.path.append(r"/mnt/c/Deepfake Image DEtection/models/build_hybrid_CNN.py")
from models.build_hybrid_CNN import build_hybrid_cnn as build_model

IMG_SIZE = (128,128)

print("\n Loading Hybrid CNN model...")
model = build_model(input_shape=IMG_SIZE + (3,))
model.summary()

os.makedirs("saved_model", exist_ok=True)
with open("saved_model/model_architecture.json", "w") as f:
    f.write(model.to_json())

# 4️ DATASET LOADING

DATASET_PATH = "/mnt/c/Deepfake Image DEtection/Final_Dataset"
BATCH_SIZE = 20
AUTOTUNE = tf.data.AUTOTUNE
try:
    IMG_SIZE = (128,128)
    BATCH = BATCH_SIZE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "validation"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int"
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

except Exception as e:
    print("\n DATASET LOADING FAILED")
    print(str(e))
    traceback.print_exc()
    exit()

# 5️ CLASS WEIGHTS (for 141K images)

train_labels = np.concatenate([y.numpy() for _, y in train_ds])
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print("\n Class Weights:", class_weights)

# 6️ CALLBACKS

def create_callbacks(name):
    os.makedirs(f"/mnt/c/Deepfake Image DEtection/checkpoints/{name}", exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            f"checkpoints/{name}/best_model.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

# 7️ GRAPH SAVING AFTER EVERY EPOCH

os.makedirs("/mnt/c/Deepfake Image DEtection/training_graphs/per_epoch", exist_ok=True)

def save_graphs(hist, epoch, phase):
    acc = hist["accuracy"]
    val_acc = hist["val_accuracy"]
    loss = hist["loss"]
    val_loss = hist["val_loss"]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.title(f"{phase} Accuracy (Epoch {epoch+1})")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.title(f"{phase} Loss (Epoch {epoch+1})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"/mnt/c/Deepfake Image DEtection/training_graphs/per_epoch/{phase}_epoch_{epoch+1}.png")
    plt.close()

class GraphSaver(keras.callbacks.Callback):
    def __init__(self, phase):
        self.phase = phase
        self.hist = {}

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.hist.setdefault(k, []).append(float(v))

        save_graphs(self.hist, epoch, self.phase)

# 8️ COMPILE MODEL

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 9️ PHASE 1 — INITIAL TRAINING (15 epochs)

INIT_EPOCHS = 15
print("\n STARTING INITIAL TRAINING...\n")

init_cb = create_callbacks("initial") + [GraphSaver("initial")]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INIT_EPOCHS,
    callbacks=init_cb,
    class_weight=class_weights
)

model.save("/mnt/c/Deepfake Image DEtection/saved_model/initial_model.h5")

#  10 PHASE 2 — FINE TUNE (5 epochs, unfreeze 20 layers)

print("\n STARTING FINE TUNING...\n")

best_init = "/mnt/c/Deepfake Image DEtection/checkpoints/initial/best_model.h5"
model = keras.models.load_model(best_init)

# SAFE for batch=20 and RTX 3050 → Unfreeze last 20 layers
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

FT_EPOCHS = 5
ft_cb = create_callbacks("finetune") + [GraphSaver("finetune")]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FT_EPOCHS,
    callbacks=ft_cb,
    class_weight=class_weights
)

model.save("/mnt/c/Deepfake Image DEtection/saved_model/final_finetuned_model.h5")

# 1️1 TEST EVALUATION

print("\n Evaluating on test set...")
results = model.evaluate(test_ds, return_dict=True)
json.dump(results, open("/mnt/c/Deepfake Image DEtection/saved_model/test_results.json","w"))
print("\nTEST RESULTS:", results)

print("\n TRAINING FINISHED SUCCESSFULLY!")
print(" Best initial model → checkpoints/initial/best_model.h5")
print(" Best fine-tuned model → checkpoints/finetune/best_model.h5")
print("Best Per-epoch graphs → training_graphs/per_epoch/")
print("Final model → saved_model/final_finetuned_model.h5")