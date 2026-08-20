from ultralytics import YOLO
import os

models = [
    "runs/detect/tomato_yolo_more_epochs/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-9/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-8/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-6/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-4/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-3/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-2/weights/best.pt",
    "yolov8n.pt"
]

images = [
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg",
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png",
    r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg"
]

for m_path in models:
    print(f"\n=====================================")
    print(f"Testing Model: {m_path}")
    try:
        model = YOLO(m_path)
        print(f"Classes: {model.names}")
        for img_p in images:
            bname = os.path.basename(img_p)
            results = model.predict(source=img_p, conf=0.10, verbose=False)
            boxes = results[0].boxes
            print(f"  Img: {bname} -> Detections: {len(boxes)}")
            if len(boxes) > 0:
                cls_counts = {}
                for b in boxes:
                    c_id = int(b.cls[0])
                    c_name = model.names[c_id]
                    cls_counts[c_name] = cls_counts.get(c_name, 0) + 1
                print(f"    Classes: {cls_counts}")
    except Exception as e:
        print(f"Error loading {m_path}: {e}")
