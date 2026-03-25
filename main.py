from ultralytics import YOLO
import cv2

# Load YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, etc.)
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model(frame)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
