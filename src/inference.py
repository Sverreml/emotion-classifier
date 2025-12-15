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
    emotions = ["Angry", "Fear", "Happy", "Sad", "Suprise"]
    predictions = model.predict(image)
    return predictions[0], emotions[tf.argmax(predictions, axis=1)[0]]

if __name__ == "__main__":
    # Example usage
    model = loadModel("Models/final_classifier_1.0.0.keras")
    sample_image = "Data/Pictures/Angry/pexels-olly-3812754.jpg"
    image = tf.io.read_file(sample_image)
    image = tf.image.decode_png(image, channels=3)
    predictions, predicted_class = predict(model, image)
    print("Predictions:", predictions)
    print("Predicted class:", predicted_class)

