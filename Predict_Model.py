import cv2
from deepface import DeepFace
import time
import numpy as np

def main():
    # Open webcam (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' or ESC to quit...")

    prev_time = 0
    emotion_window = []  # Store recent emotions for smoothing
    window_size = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            # Analyze emotion using DeepFace with opencv backend (more stable)
            result = DeepFace.analyze(
                img_path=np.asarray(frame),
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',  # Changed to opencv from mediapipe
                silent=True  # Suppress verbose output
            )

            # Handle result (can be list or dict)
            if isinstance(result, list):
                analysis = result[0] if len(result) > 0 else None
            else:
                analysis = result

            if analysis:
                dominant_emotion = analysis.get('dominant_emotion')
                emotion_scores = analysis.get('emotion', {})
                confidence = emotion_scores.get(dominant_emotion, 0)
                region = analysis.get('region', {})

                # Smooth emotion results using a moving window
                emotion_window.append(dominant_emotion)
                if len(emotion_window) > window_size:
                    emotion_window.pop(0)
                
                # Get most frequent emotion in window
                if emotion_window:
                    smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
                else:
                    smoothed_emotion = None

                if smoothed_emotion and confidence >= 60:  # Lowered confidence threshold
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display emotion and confidence
                    label = f'{smoothed_emotion}: {confidence:.1f}%'
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)

                    print(f"Emotion: {smoothed_emotion}, Confidence: {confidence:.1f}%")

        except Exception as e:
            print(f"Detection error: {str(e)}")
            continue

        # Calculate and show FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Webcam Emotion Detection", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()