import keras
from keras import layers, regularizers
from config import IMAGE_SIZE, NUM_CLASSES

def build_classifier():
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(
            filters, 3, padding="same",
            kernel_regularizer=regularizers.l2(1e-5)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)
