from ultralytics import YOLO
import cv2
import os

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787065223962.png"
orig_img = cv2.imread(img_path)
h, w = orig_img.shape[:2]
print(f"Image Resolution: {w}x{h}")

model_paths = [
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-9\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\tomato_yolo_more_epochs\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-6\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-8\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\yolov8n.pt",
]

for mp in model_paths:
    if not os.path.exists(mp):
        continue
    model = YOLO(mp)
    m_name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
    print(f"\n================ Model: {m_name} ================")
    print("Classes:", model.names)
    
    for sz in [640, 1024, 1280]:
        for conf in [0.25, 0.15]:
            results = model(orig_img, imgsz=sz, conf=conf, verbose=False)[0]
            boxes = results.boxes
            labels = [f"{model.names[int(b.cls[0])]}:{float(b.conf[0]):.2f}" for b in boxes]
            print(f"  imgsz={sz}, conf={conf}: {len(boxes)} boxes -> {labels[:15]}")
