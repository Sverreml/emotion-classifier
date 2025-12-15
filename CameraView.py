import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained emotion detection model
model = load_model("Models/final_classifier_1.0.0.keras")
emotion_labels = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# Load face cascade classifier once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the region of interest (the face)
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (180, 180))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion
            predictions = model.predict(roi, verbose=0)[0]
            emotion_probability = np.max(predictions)
            label = emotion_labels[np.argmax(predictions)]

            # Display the label and probability on the frame
            cv2.putText(frame, f"{label}: {emotion_probability:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()