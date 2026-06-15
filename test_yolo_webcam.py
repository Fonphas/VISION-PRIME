import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    annotated = results.plot()

    cv2.imshow("YOLO11n - CUDA Test (press Q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()