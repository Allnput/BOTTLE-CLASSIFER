import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow import keras

# ------------------------------
# Load CNN models
# ------------------------------
cnn_air = keras.models.load_model("model_water_level.keras")
labels_air = ["Penuh", "Kurang", "Overflow"]

cnn_defect = keras.models.load_model("model_condition_botle.keras")
labels_defect = ["Proper", "Defect"]

# ------------------------------
# Load YOLO model tutup botol 2-class
# ------------------------------
yolo_cap = YOLO("model_cap_bottley.pt")
cap_labels = ["Proper Cap", "Missing/Defect Cap"]
conf_threshold = 0.3

# ------------------------------
# Load YOLO botol (hanya untuk deteksi bounding box)
# ------------------------------
yolo_bottle = YOLO("yolov8n.pt")
bottle_class_id = 39

# ------------------------------
# Buka kamera
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO botol untuk bounding box
    results_bottle = yolo_bottle.predict(frame)[0]

    for i, box in enumerate(results_bottle.boxes.xyxy):
        cls_id = int(results_bottle.boxes.cls[i].item())
        conf_box = results_bottle.boxes.conf[i].item()
        if cls_id != bottle_class_id or conf_box < conf_threshold:
            continue

        # Bounding box aman
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        img = cv2.resize(crop, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred_air = cnn_air.predict(img, verbose=0)
        label_air = labels_air[np.argmax(pred_air)]
        conf_air = np.max(pred_air) * 100

        pred_defect = cnn_defect.predict(img, verbose=1)
        label_defect = labels_defect[np.argmax(pred_defect)]
        conf_defect = np.max(pred_defect) * 100

        results_cap = yolo_cap.predict(crop)[0]
        label_cap = "Unknown"
        conf_cap = 0
        if len(results_cap.boxes) > 0:
            cls_id_cap = int(results_cap.boxes.cls[0].item())
            conf_cap = results_cap.boxes.conf[0].item() * 100
            if cls_id_cap < len(cap_labels):
                label_cap = cap_labels[cls_id_cap]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame, f"Cap: {label_cap} ({conf_cap:.1f}%)",
                    (x1, max(0, y1-25)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

        cv2.putText(frame, f"Botol: {label_defect} ({conf_defect:.1f}%)",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 2)

        # Label level air → di kanan bounding box
        cv2.putText(frame, f"Air: {label_air} ({conf_air:.1f}%)",
                    (x2 + 10, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Botol + Cap + Air", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()