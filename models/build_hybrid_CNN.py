# models/hybrid_cnn.py
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import layers, models, regularizers

def build_hybrid_cnn(input_shape=(128, 128, 3)):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False

    #  Keep everything connected in one graph
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    
    #  Use base_model directly — do NOT wrap it in a separate submodel
    x = base_model(x, training=False)

    # Continue with your head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model