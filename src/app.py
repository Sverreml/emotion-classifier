from fastapi import FastAPI
from model import __version__
import inference

app = FastAPI()
model_path = f"Models\\final_classifier{__version__}.keras"
model = inference.loadModel(model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification API"}

@app.post("/predict/")
def predict_image(image: bytes):
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    from io import BytesIO

    # Load image from bytes
    img = Image.open(BytesIO(image)).convert("RGB")
    img = np.array(img)

    # Get predictions
    predictions, predicted_class = inference.predict(model, img)
    predicted_class = predicted_class.numpy()[0]
    confidence = tf.nn.softmax(predictions[0])[predicted_class].numpy()

    return {
        "predicted_class": int(predicted_class),
        "confidence": float(confidence)
    }