from fastapi import FastAPI, UploadFile, File, HTTPException
from config import VERSION
import inference
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load model on startup
try:
    model_path = f"Models/final_classifier_{VERSION}.keras"
    model = inference.loadModel(model_path)
except Exception as e:
    print(f"Warning: Could not load model at {model_path}: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Emotion Classification API",
        "version": VERSION,
        "emotions": ["Angry", "Fear", "Happy", "Sad", "Surprise"]
    }

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded image.
    
    Returns:
        {
            "emotion": str,
            "confidence": float,
            "all_predictions": dict
        }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
        
        # Get predictions
        predictions, predicted_emotion = inference.predict(model, img_array)
        
        # Get confidence for the predicted emotion
        softmax_predictions = tf.nn.softmax(predictions).numpy()
        max_confidence = float(np.max(softmax_predictions))
        
        # Map all predictions to emotion labels
        emotions = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
        all_predictions = {
            emotion: float(conf) 
            for emotion, conf in zip(emotions, softmax_predictions)
        }
        
        return {
            "emotion": predicted_emotion.numpy() if hasattr(predicted_emotion, 'numpy') else predicted_emotion,
            "confidence": max_confidence,
            "all_predictions": all_predictions
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )