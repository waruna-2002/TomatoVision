import cv2
import numpy as np
from ultralytics import YOLO

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787065223962.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 1. Color check function for Tomato Verification
def is_valid_tomato_color(crop_bgr):
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Check if purple/violet cabbage (H in 130-165, S > 40)
    purple_pixels = np.sum((h_channel >= 130) & (h_channel <= 165) & (s_channel > 40))
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if (purple_pixels / total_pixels) > 0.25:
        return False # REJECT: It's purple cabbage!
        
    # Check if brown/potato (low saturation, low brightness, gray/brown)
    # Check if tomato color (Red/Orange/Yellow/Green-yellow)
    red1 = (h_channel <= 15) & (s_channel > 50) & (v_channel > 50)
    red2 = (h_channel >= 165) & (s_channel > 50) & (v_channel > 50)
    orange_yellow = (h_channel > 15) & (h_channel <= 45) & (s_channel > 50) & (v_channel > 50)
    green_tomato = (h_channel > 45) & (h_channel <= 85) & (s_channel > 50) & (v_channel > 50)
    
    tomato_pixels = np.sum(red1 | red2 | orange_yellow | green_tomato)
    return (tomato_pixels / total_pixels) >= 0.20

# 2. Test YOLO models with Color Validation + Multi-Scale Detection
m_path = r"d:\project\TomatoVision\tomatovision-ml\runs\detect\tomato_yolo_more_epochs\weights\best.pt"
model = YOLO(m_path)

# Run with imgsz=1280
results = model(img, imgsz=1280, conf=0.10, iou=0.45)[0]

valid_detections = []
rejected = 0

for b in results.boxes:
    xyxy = [int(v) for v in b.xyxy[0].tolist()]
    x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
    crop = img[y1:y2, x1:x2]
    
    cls_id = int(b.cls[0])
    raw_name = model.names[cls_id]
    conf = float(b.conf[0])
    
    if is_valid_tomato_color(crop):
        # Determine genuine ripeness based on color & model
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv_crop[:, :, 0])
        mean_s = np.mean(hsv_crop[:, :, 1])
        mean_v = np.mean(hsv_crop[:, :, 2])
        
        if (mean_h <= 14 or mean_h >= 165) and mean_s > 60:
            stage = "fresh"
        elif 15 <= mean_h <= 42:
            stage = "overripe" # orange / yellow-red
        elif 43 <= mean_h <= 85:
            stage = "unripe" # green / breaker
        else:
            stage = "fresh"
            
        valid_detections.append((stage, conf, (x1, y1, x2, y2)))
    else:
        rejected += 1
        print(f"[REJECTED NON-TOMATO FALSE POSITIVE]: {raw_name} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")

print(f"\nTotal Valid Tomato Detections: {len(valid_detections)}")
print(f"Total Rejected False Positives (e.g. Cabbage/Potatoes): {rejected}")
for d in valid_detections[:10]:
    print(f"  -> Tomato: {d[0]} (conf {d[1]:.2f}) at {d[2]}")
