from ultralytics import YOLO

img_path = r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg"
m_path = r"d:\project\TomatoVision\tomatovision-ml\runs\detect\tomato_yolo_more_epochs\weights\best.pt"

model = YOLO(m_path)
results = model(img_path, conf=0.15, iou=0.45)[0]

counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for b in results.boxes:
    cls_id = int(b.cls[0])
    raw_name = model.names[cls_id].lower()
    score = float(b.conf[0])
    if raw_name in ["ripe", "fresh"]:
        counts["fresh"] += 1
    elif raw_name in ["overipe", "overripe"]:
        counts["overripe"] += 1
    elif raw_name in ["unripe"]:
        counts["unripe"] += 1
    elif raw_name in ["spoiled"]:
        counts["spoiled"] += 1
    print(f"Detected: {raw_name} ({score:.2f})")

print(f"Total detected: {len(results.boxes)}")
print(f"Counts: {counts}")
