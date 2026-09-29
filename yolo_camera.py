from ultralytics import YOLO
import cv2

# YOLO model load
model = YOLO("yolov8n.pt")

# Laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha!")
    exit()

print("AI Camera Started - Q dabakar band karo")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera se frame nahi mil raha!")
        break

    # AI detection
    results = model(frame)

    # Detection result screen par show
    annotated_frame = results[0].plot()

    cv2.imshow("AI Fire Detection", annotated_frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()