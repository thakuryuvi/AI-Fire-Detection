import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera open nahi ho raha!")
    exit()

print("Camera started. Band karne ke liye Q dabao.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera se frame nahi mil raha!")
        break

    cv2.imshow("Fire Detection - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()