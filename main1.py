from ultralytics import YOLO
import cv2
import numpy as np
import requests

# ESP32-CAM IP address (shown in Serial Monitor)
ESP32_IP = "10.105.6.23"   # change this

# URLs
CAM_URL = f"http://{ESP32_IP}/cam-hi.jpg"
SEND_URL = f"http://{ESP32_IP}/send?data="

# Load YOLO model
model = YOLO("best.pt")
print("Model Classes:")
for i, name in model.names.items():
    print(f"{i} : {name}")

while True:
    
    # Get image from ESP32-CAM
    try:
        img_resp = requests.get(CAM_URL, timeout=2)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
    except:
        print("Camera fetch failed")
        continue

    # Run YOLO detection
    results = model(frame)

    detected_label = None

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                detected_label = model.names[cls]

    # Send detection to ESP32
    if detected_label is not None:
        try:
            requests.get(SEND_URL + detected_label)
            print("Sent to ESP32:", detected_label)
        except:
            print("Send failed")

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("ESP32 YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
