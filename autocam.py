import cv2
import os
import time

# === CONFIGURATION ===
label = "neutral"  # Change this to the label you want to capture
base_dir = r"C:\Users\user\Desktop\MODELTRAINING"

# Create label folder in the specified directory
save_dir = os.path.join(base_dir, label)
os.makedirs(save_dir, exist_ok=True)

# === INITIALIZE CAMERA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print(f"[INFO] Capturing images for label '{label}' for 5 seconds...")

start_time = time.time()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Save frame
    img_path = os.path.join(save_dir, f"{label}_{count+1}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1

    # Display frame
    cv2.imshow("Capturing...", frame)

    # Stop after 5 seconds
    if time.time() - start_time > 5:
        break

    # Optional: press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Saved {count} images in '{save_dir}'")
