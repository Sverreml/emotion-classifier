import tensorflow as tf
import keras

IMAGE_SIZE = (180, 180)
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# ----------------------------
# Model loading
# ----------------------------

def load_model(path):
    return keras.models.load_model(path)


# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_image(image):
    """
    image: Tensor [H, W, 3], dtype uint8 or float32
    returns: Tensor [1, 180, 180, 3]
    """
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.expand_dims(image, axis=0)
    return image


def load_image_from_path(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return image


# ----------------------------
# Prediction
# ----------------------------

def predict(model, image):
    image = preprocess_image(image)
    probs = model(image, training=False)[0]

    class_idx = tf.argmax(probs).numpy()
    confidence = probs[class_idx].numpy()

    return {
        "class_index": class_idx,
        "class_name": CLASS_NAMES[class_idx],
        "confidence": float(confidence),
        "probabilities": probs.numpy()
    }


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    model = load_model("Models/final_classifier_1.0.0.keras")

    image = load_image_from_path(
        "Data/Pictures/Angry/pexels-olly-3812754.jpg"
    )

    result = predict(model, image)

    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("Probabilities:", result["probabilities"])
