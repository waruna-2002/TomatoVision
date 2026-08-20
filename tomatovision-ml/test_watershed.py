import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787065223962.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# 1. Target Tomato Colors (Red, Orange, Yellow, Breaker Green)
mask_red1 = (h_chan <= 18) & (s_chan > 50) & (v_chan > 50)
mask_red2 = (h_chan >= 162) & (s_chan > 50) & (v_chan > 50)
mask_orange = (h_chan > 18) & (h_chan <= 38) & (s_chan > 60) & (v_chan > 60)
mask_green = (h_chan > 38) & (h_chan <= 80) & (s_chan > 50) & (v_chan > 50)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_green).astype(np.uint8) * 255

# Exclude purple/violet (cabbage) completely
cabbage_mask = (h_chan >= 125) & (h_chan <= 160) & (s_chan > 40)
tomato_mask[cabbage_mask] = 0

# Exclude bottom road / gravel (low saturation or gray)
gray_mask = (s_chan < 35) | (v_chan < 35)
tomato_mask[gray_mask] = 0

# 2. Distance Transform + Peak Finding for Individual Tomatoes in Crate
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Threshold distance transform to find individual tomato centers
_, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Connected components on foreground seeds
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

detected = []
output_img = img.copy()

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 15: # Ignore tiny noise
        continue
        
    cx, cy = centroids[i]
    # Estimate radius from distance transform
    r = int(dist_transform[int(cy), int(cx)] * 2.2)
    r = max(14, min(45, r)) # Constrain to realistic tomato size in crate
    
    x1 = max(0, int(cx - r))
    y1 = max(0, int(cy - r))
    x2 = min(w, int(cx + r))
    y2 = min(h, int(cy + r))
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue
        
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv_crop[:, :, 0])
    mean_s = np.mean(hsv_crop[:, :, 1])
    mean_v = np.mean(hsv_crop[:, :, 2])
    
    # Classify Ripeness
    if (mean_h <= 14 or mean_h >= 165) and mean_s > 60:
        stage = "fresh" # Ripe Red
        conf = 0.94
        color = (0, 255, 0)
    elif 15 <= mean_h <= 36:
        stage = "overripe" # Orange / Turning
        conf = 0.91
        color = (0, 200, 255)
    elif 37 <= mean_h <= 85:
        stage = "unripe" # Green / Breaker
        conf = 0.93
        color = (255, 200, 0)
    else:
        stage = "fresh"
        conf = 0.88
        color = (0, 255, 0)
        
    detected.append({
        "class_name": stage,
        "confidence": conf,
        "box": [x1, y1, x2, y2]
    })
    
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output_img, f"{stage}", (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

print(f"Total Individual Tomatoes Detected in Crate: {len(detected)}")
counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for d in detected:
    counts[d["class_name"]] += 1
print(f"Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/watershed_detected.jpg", output_img)
