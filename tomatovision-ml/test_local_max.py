import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787074656565.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# 1. Broad Crate Tomato Region Isolation
core_red1 = (h_chan <= 15) & (s_chan > 45) & (v_chan > 40)
core_red2 = (h_chan >= 162) & (s_chan > 45) & (v_chan > 40)
core_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 55) & (v_chan > 45)
core_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 55) & (v_chan > 45)
core_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 45) & (v_chan > 40)

core_mask = (core_red1 | core_red2 | core_orange | core_yellow | core_unripe).astype(np.uint8) * 255
core_mask[h_chan > 55] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 40] = 0

kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
cluster_mask = cv2.dilate(core_mask, kernel_group, iterations=2)

contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    main_cluster = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cluster)
    rx1 = max(0, rx - 15)
    ry1 = max(0, ry - 15)
    rx2 = min(w, rx + rw + 15)
    ry2 = min(h, ry + rh + 15)
else:
    rx1, ry1, rx2, ry2 = 0, 0, w, h

# 2. Fruit Mask inside Crate
roi_img = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

roi_red1 = (roi_h <= 15) & (roi_s > 40) & (roi_v > 35)
roi_red2 = (roi_h >= 162) & (roi_s > 40) & (roi_v > 35)
roi_orange = (roi_h > 15) & (roi_h <= 28) & (roi_s > 45) & (roi_v > 40)
roi_yellow = (roi_h > 28) & (roi_h <= 38) & (roi_s > 45) & (roi_v > 40)
roi_unripe = (roi_h > 38) & (roi_h <= 55) & (roi_s > 40) & (roi_v > 35)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 55] = 0
roi_mask[roi_s <= 40] = 0

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_clean)
dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

# 3. Local Maxima using Morphological Dilation Peak Finding
# A pixel is a local maximum if dist == dilate(dist)
kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
dilated = cv2.dilate(dist, kernel_peak)
local_max = (dist == dilated) & (dist > 3.0) & (dist > 0.08 * dist.max())

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))

tomatoes = []
output_img = img.copy()

for i in range(1, num_labels):
    cx, cy = centroids[i]
    cx_i, cy_i = int(cx), int(cy)
    r = int(dist[cy_i, cx_i] * 1.7)
    r = max(10, min(36, r))
    
    gx1 = max(0, int(rx1 + cx - r))
    gy1 = max(0, int(ry1 + cy - r))
    gx2 = min(w, int(rx1 + cx + r))
    gy2 = min(h, int(ry1 + cy + r))
    
    crop = img[gy1:gy2, gx1:gx2]
    if crop.size == 0:
        continue
        
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    # Check spoilage / dark spots
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0
    
    if dark_ratio > 0.28:
        stage = "spoiled"
        color = (255, 71, 87)
    elif (mean_h <= 13 or mean_h >= 164):
        stage = "ripe"
        color = (46, 213, 115)
    elif 14 <= mean_h <= 26:
        stage = "overripe"
        color = (255, 159, 67)
    elif 27 <= mean_h <= 55:
        stage = "unripe"
        color = (0, 210, 211)
    else:
        stage = "ripe"
        color = (46, 213, 115)
        
    tomatoes.append(stage)
    cv2.rectangle(output_img, (gx1, gy1), (gx2, gy2), color, 2)

counts = {
    "ripe": tomatoes.count("ripe"),
    "unripe": tomatoes.count("unripe"),
    "overripe": tomatoes.count("overripe"),
    "spoiled": tomatoes.count("spoiled")
}

print(f"Local Maxima Peak Finding -> Total Detected in Crate: {len(tomatoes)}")
print(f"4 Category Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/local_max_detected.jpg", output_img)
