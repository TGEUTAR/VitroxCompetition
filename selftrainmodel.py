import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model(r'C:\Users\user\Desktop\MODELTRAINING\emotion_model.h5')

# Labels for your 3 classes
emotion_labels = ['Happy', 'Sad', 'Neutral']

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_color = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_color, (48, 48))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        
        preds = model.predict(face_input, verbose=0)
        emotion_index = np.argmax(preds)
        confidence = preds[0][emotion_index]
        
        if confidence > 0.6:
            label = f"{emotion_labels[emotion_index]}: {confidence*100:.1f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
