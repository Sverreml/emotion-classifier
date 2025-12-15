import tensorflow as tf
import keras

def preprocess_image(image):
    """Resize and preprocess image for prediction"""
    import tensorflow as tf
    # Resize to 180x180 if needed
    if image.shape[0] != 180 or image.shape[1] != 180:
        image = tf.image.resize(image, [180, 180])
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    return image

def loadModel(path):
    return keras.saving.load_model(
        path,
        custom_objects=None,
        compile=True,
        safe_mode=True
    )

def predict(model, image):
    image = preprocess_image(image)
    return (model.predict(image), tf.argmax(model.predict(image), axis=1))

if __name__ == "__main__":
    # Example usage
    model = loadModel("Models\\final_classifier.keras")
    sample_image = tf.random.uniform((200, 200, 3))  # Example image
    predictions = predict(model, sample_image)
    print("Predictions:", predictions)
    loss = keras.losses.SparseCategoricalCrossentropy()
    predicted_class = tf.argmax(predictions, axis=1)
    print("Predicted class:", predicted_class.numpy())
    print("Loss value:", loss([1], predictions).numpy())
    model, image = None, None  # Free up resources
    tf.keras.backend.clear_session()
    print("Resources cleared.")
    print("Inference complete.")

