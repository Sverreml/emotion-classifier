import keras
from keras import layers
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
import pathlib


__version__ = "1.0.0"
base_dir = pathlib.Path(r"Data/Sets")

print("Loading trainingset.")

train_dataset = image_dataset_from_directory(
    base_dir / "train",
    image_size=(180, 180),
    batch_size=32,
    seed=42,
)

print("Loading validationset.")

validation_dataset = image_dataset_from_directory(
    base_dir / "validation",
    image_size=(180, 180),
    batch_size=32,
    seed=42
)

print("Loading testset.")

test_dataset = image_dataset_from_directory(
    base_dir / "test",
    image_size=(180, 180),
    batch_size=32,
    seed=42
)




def classifier():
    inputs = keras.Input(shape=(180, 180, 3))

    x = layers.Rescaling(1./255)(inputs)

    # Block 1
    x = layers.Conv2D(
        32, 3, padding="same",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(
        64, 3, padding="same",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(
        128, 3, padding="same",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 4
    x = layers.Conv2D(
        256, 3, padding="same",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-5)
    )(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(5, activation="softmax")(x)

    return keras.Model(inputs, outputs)
    
model = classifier()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="Models/stabilized_model.keras",
        save_best_only=True,
        monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )
]

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    train_dataset,
    epochs = 100,
    validation_data = validation_dataset,
    callbacks = callbacks,
    verbose = 2
)





accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

best_model = keras.models.load_model("Models/stabilized_model.keras")
test_loss, test_acc = best_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")
model.save(f"Models/final_classifier{__version__}.keras")