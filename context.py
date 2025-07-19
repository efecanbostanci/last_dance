import cv2
import numpy as np
from ultralytics import YOLO


person_model = YOLO("yolov8n.pt")     
mask_model = YOLO("best.pt")          

# Video dosyasını aç
video_path = "video1.mp4"
cap = cv2.VideoCapture("video2.mp4")

# Ayarlar
DIST_THRESHOLD = 100
unique_ids = set()

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. İnsan takibi
    person_results = person_model.track(frame, persist=True, conf=0.5, classes=0)
    if person_results[0].boxes is None:
        continue

    boxes = person_results[0].boxes
    annotated_frame = person_results[0].plot()

    
    people_coords = []

    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        for person_id in ids:
            unique_ids.add(person_id)

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        people_coords.append(((center_x, center_y), (x1, y1, x2, y2)))

    # 2. Sosyal mesafe kontrolü
    for i in range(len(people_coords)):
        (p1, box1) = people_coords[i]
        color = (0, 255, 0)  # varsayılan yeşil
        for j in range(i + 1, len(people_coords)):
            (p2, box2) = people_coords[j]
            dist = euclidean_distance(p1, p2)
            if dist < DIST_THRESHOLD:
                color = (0, 0, 255)  # ihlal → kırmızı
                cv2.line(annotated_frame, p1, p2, (0, 0, 255), 2)

        x1, y1, x2, y2 = box1
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated_frame, p1, 3, color, -1)

   
    mask_results = mask_model(frame, conf=0.5)

    if mask_results[0].boxes:
        for box in mask_results[0].boxes:
            cls_id = int(box.cls[0])
            label = mask_results[0].names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "mask":
                color = (0, 255, 0)
            elif label == "no_mask":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

  
    cv2.putText(annotated_frame, f"Toplam Kişi: {len(unique_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv2.imshow("Sosyal Mesafe + Maske Tespiti", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Videodaki toplam kişi sayısı: {len(unique_ids)}")
