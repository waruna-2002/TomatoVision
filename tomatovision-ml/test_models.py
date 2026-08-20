import os
from ultralytics import YOLO

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787064846656.png"

models = [
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-2\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-3\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-4\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-6\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-8\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\models\tomato_yolo_run-9\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\runs\detect\tomato_yolo_more_epochs\weights\best.pt",
    r"d:\project\TomatoVision\tomatovision-ml\yolov8n.pt"
]

print("Testing user image against all trained models:")
for m_path in models:
    if not os.path.exists(m_path):
        continue
    try:
        model = YOLO(m_path)
        m_name = os.path.basename(os.path.dirname(os.path.dirname(m_path)))
        print(f"\n--- Model: {m_name} ---")
        print("Classes:", model.names)
        
        for conf in [0.25, 0.15, 0.05]:
            results = model(img_path, conf=conf, verbose=False)[0]
            detected = []
            for b in results.boxes:
                cls_id = int(b.cls[0])
                c_name = model.names[cls_id]
                score = float(b.conf[0])
                detected.append(f"{c_name} ({score:.2f})")
            print(f"  Conf >= {conf}: {len(detected)} detected -> {detected[:8]}")
    except Exception as e:
        print(f"Error on {m_path}: {e}")
