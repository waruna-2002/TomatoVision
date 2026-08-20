import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask_red1 = (h_chan <= 15) & (s_chan > 45) & (v_chan > 40)
mask_red2 = (h_chan >= 162) & (s_chan > 45) & (v_chan > 40)
mask_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 55) & (v_chan > 45)
mask_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 55) & (v_chan > 45)
mask_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 45) & (v_chan > 40)

core_mask = (mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_unripe).astype(np.uint8) * 255
core_mask[h_chan > 55] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 40] = 0

# Morphological opening to merge connected parts of single fruit
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel_close)

contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_tomatoes = []
output = img.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area < 1000:
        continue
    x, y, cw, ch = cv2.boundingRect(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    aspect_ratio = float(cw) / ch if ch > 0 else 0
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # REJECT Table strips and background rectangles:
    # 1. Long horizontal strips (AR > 2.3 or AR < 0.4)
    # 2. Non-organic flat borders (Circularity < 0.35)
    # 3. High width spanning > 60% of frame while thin (ch < 0.15 * h)
    if aspect_ratio > 2.2 or aspect_ratio < 0.45:
        print(f"REJECTED non-tomato strip: AR={aspect_ratio:.2f}, BBox=[{x},{y},{cw},{ch}]")
        continue
    if circularity < 0.35:
        print(f"REJECTED non-circular object: Circ={circularity:.2f}, BBox=[{x},{y},{cw},{ch}]")
        continue
    if cw > 0.60 * w and ch < 0.18 * h:
        print(f"REJECTED table border: BBox=[{x},{y},{cw},{ch}]")
        continue
        
    crop = img[y:y+ch, x:x+cw]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    if (mean_h <= 13 or mean_h >= 164):
        stage = "ripe"
    elif 14 <= mean_h <= 26:
        stage = "overripe"
    elif 27 <= mean_h <= 55:
        stage = "unripe"
    else:
        stage = "ripe"
        
    valid_tomatoes.append((stage, [x, y, x+cw, y+ch]))
    cv2.rectangle(output, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
    cv2.putText(output, stage, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"\nFinal Tomatoes Detected in Single-Fruit Image: {len(valid_tomatoes)}")
for t in valid_tomatoes:
    print(f"  Stage: {t[0]}, Box: {t[1]}")
