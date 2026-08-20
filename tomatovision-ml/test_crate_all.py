import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# STAGE 1: Broad Tomato Cluster ROI with edge tolerance
core_red1 = (h_chan <= 15) & (s_chan > 75) & (v_chan > 60)
core_red2 = (h_chan >= 164) & (s_chan > 75) & (v_chan > 60)
core_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 85) & (v_chan > 70)
core_unripe = (h_chan > 28) & (h_chan <= 48) & (s_chan > 85) & (v_chan > 70)

core_mask = (core_red1 | core_red2 | core_orange | core_unripe).astype(np.uint8) * 255
core_mask[h_chan > 48] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 70] = 0

kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
cluster_mask = cv2.dilate(core_mask, kernel_group, iterations=2)

contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    main_cluster = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cluster)
    rx1 = max(0, rx - 18)
    ry1 = max(0, ry - 18)
    rx2 = min(w, rx + rw + 18)
    ry2 = min(h, ry + rh + 18)
else:
    rx1, ry1, rx2, ry2 = 0, 0, w, h

print(f"Isolated Crate ROI: X=[{rx1}, {rx2}], Y=[{ry1}, {ry2}]")

# STAGE 2: Dense Fruit Segmentation
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

roi_red1 = (roi_h <= 14) & (roi_s > 65) & (roi_v > 55)
roi_red2 = (roi_h >= 164) & (roi_s > 65) & (roi_v > 55)
roi_orange = (roi_h > 14) & (roi_h <= 27) & (roi_s > 75) & (roi_v > 65)
roi_unripe = (roi_h > 27) & (roi_h <= 45) & (roi_s > 75) & (roi_v > 65)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 45] = 0
roi_mask[roi_s <= 65] = 0

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_clean)
dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(dist, 0.22 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

tomatoes = []
output_img = img.copy()

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < 6:
        continue
    cx, cy = centroids[i]
    r = int(dist[int(cy), int(cx)] * 1.8)
    r = max(10, min(40, r))
    
    gx1 = max(0, int(rx1 + cx - r))
    gy1 = max(0, int(ry1 + cy - r))
    gx2 = min(w, int(rx1 + cx + r))
    gy2 = min(h, int(ry1 + cy + r))
    
    crop = img[gy1:gy2, gx1:gx2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    # Classify into exact 4 categories: Ripe, Unripe, Overripe, Spoiled
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0
    
    if dark_spots > 0.40:
        stage = "spoiled"
        color = (255, 71, 87)
    elif (mean_h <= 13 or mean_h >= 164):
        stage = "ripe"
        color = (46, 213, 115)
    elif 14 <= mean_h <= 25:
        stage = "overripe"
        color = (255, 159, 67)
    elif 26 <= mean_h <= 45:
        stage = "unripe"
        color = (0, 210, 211)
    else:
        stage = "ripe"
        color = (46, 213, 115)
        
    tomatoes.append({
        "stage": stage,
        "box": [gx1, gy1, gx2, gy2]
    })
    cv2.rectangle(output_img, (gx1, gy1), (gx2, gy2), color, 2)
    cv2.putText(output_img, stage, (gx1, max(12, gy1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

counts = {
    "ripe": sum(1 for t in tomatoes if t["stage"] == "ripe"),
    "unripe": sum(1 for t in tomatoes if t["stage"] == "unripe"),
    "overripe": sum(1 for t in tomatoes if t["stage"] == "overripe"),
    "spoiled": sum(1 for t in tomatoes if t["stage"] == "spoiled")
}

print(f"Total Detected in Crate: {len(tomatoes)}")
print(f"4 Categories Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/crate_all.jpg", output_img)
