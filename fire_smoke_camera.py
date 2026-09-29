from ultralytics import YOLO
import cv2
import time
import winsound

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha!")
    exit()

last_alert_time = 0
alert_cooldown = 3

print("AI Fire & Smoke Detection Started")
print("Q dabakar band karo")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    fire_detected = False
    smoke_detected = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence >= 0.50:
            if class_id == 0:
                fire_detected = True
            elif class_id == 1:
                smoke_detected = True

    if fire_detected:
        cv2.putText(
            annotated_frame,
            "WARNING: FIRE DETECTED!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if time.time() - last_alert_time >= alert_cooldown:
            winsound.PlaySound(
                "alarm.wav",
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            last_alert_time = time.time()

    elif smoke_detected:
        cv2.putText(
            annotated_frame,
            "WARNING: SMOKE DETECTED!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if time.time() - last_alert_time >= alert_cooldown:
            winsound.PlaySound(
                "alarm.wav",
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            last_alert_time = time.time()

    else:
        cv2.putText(
            annotated_frame,
            "STATUS: SAFE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Fire & Smoke Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()