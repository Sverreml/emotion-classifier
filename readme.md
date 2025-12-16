EMOTION CLASSIFIER - Real-time Facial Emotion Recognition System  
================================================================================  

PROJECT OVERVIEW  
================================================================================  
A Python-based emotion detection system using deep learning (TensorFlow/Keras).  
Detects 5 emotions: Angry, Fear, Happy, Sad, Surprise  
  
Features:  
- Real-time emotion detection via webcam (CameraView.py)  
- FastAPI web service for image-based predictions (src/app.py)  
- CNN model architecture with batch normalization and regularization  
- Automatic data preprocessing and train/validation/test split  
- Model checkpointing and early stopping during training  


INSTALLATION & SETUP  
================================================================================  
1. Install Python dependencies:  
   pip install -r requirements.txt  

2. Ensure data structure exists:  
   Data/  
   ├── Pictures/ (raw images organized by emotion)  
   ├── Sets/  
   │   ├── train/  
   │   ├── validation/  
   │   └── test/  

3. Models folder:  
   Models/ (contains .keras model files)  


USAGE  
================================================================================  

Real-time Webcam Detection:  
  python CameraView.py  
  (Press 'q' to quit)  

Start FastAPI Server:  
  uvicorn src.app:app --reload  
  
  Endpoints:  
  - GET /  → Welcome message  
  - POST /predict/  → Upload image (bytes) for emotion prediction  

Prepare Training Data:  
  python src/preprocessing.py  
  (Splits raw images into 50% train, 25% validation, 25% test)  

Train Model:  
  python src/model.py  
  (Trains CNN for ~100 epochs with callbacks)  


PROJECT STRUCTURE  
================================================================================  

CameraView.py              - Real-time emotion detection using webcam  
src/  
  ├── app.py              - FastAPI web service  
  ├── inference.py        - Model loading and prediction utilities  
  ├── model.py            - CNN architecture and training pipeline  
  └── preprocessing.py    - Data splitting utility  

Data/  
  ├── Pictures/           - Raw images (organized by emotion)  
  └── Sets/               - Split datasets (train/val/test)  

Models/  
  ├── final_classifier_1.0.0.keras  
  └── stabilized_model.keras  


KEY COMPONENTS    
================================================================================  

Model Architecture (src/model.py):  
- 4 convolutional blocks with batch normalization  
- L2 regularization (1e-5)  
- Dropout layers (0.2)  
- Global average pooling  
- Dense head (128 units) with softmax output (5 classes)  
- Input: 180x180x3 RGB images  

Training:  
- Optimizer: Adam  
- Loss: Sparse categorical crossentropy  
- Batch size: 32  
- Max epochs: 100  
- Callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  

Inference (src/inference.py):  
- Resizes images to 180x180  
- Returns predictions and highest confidence emotion  
- Loads models with custom safety mode  


KNOWN ISSUES / TODO    
================================================================================  
- Training script needs test set evaluation at end  
- Better error handling needed in API endpoints  
- Missing requirements.txt (needs: tensorflow, keras, opencv-python, fastapi, pillow, etc.)  


REQUIREMENTS  
================================================================================  
Python 3.8+  
- tensorflow >= 2.10  
- keras  
- opencv-python  
- fastapi  
- uvicorn  
- pillow  
- numpy  
- matplotlib  
  

NOTES  
================================================================================    
- Emotion classes: Angry, Fear, Happy, Sad, Surprise  
- Model was trained on balanced dataset splits  
- Images are normalized to [0,1] range during preprocessing  
- Face detection uses OpenCV's Haar Cascade (limited robustness)  

================================================================================
