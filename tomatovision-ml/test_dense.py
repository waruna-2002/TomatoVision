import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787067683505.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# STAGE 1: Broad Tomato Cluster ROI
core_red1 = (h_chan <= 15) & (s_chan > 75) & (v_chan > 60)
core_red2 = (h_chan >= 164) & (s_chan > 75) & (v_chan > 60)
core_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 85) & (v_chan > 70)
core_unripe = (h_chan > 28) & (h_chan <= 48) & (s_chan > 85) & (v_chan > 70)

core_mask = (core_red1 | core_red2 | core_orange | core_unripe).astype(np.uint8) * 255
# Exclude outside leaves (H > 48) and cabbage (H in 120-165)
core_mask[h_chan > 48] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 70] = 0

kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
cluster_mask = cv2.dilate(core_mask, kernel_group, iterations=2)

contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    main_cluster = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cluster)
    rx1 = max(0, rx - 10)
    ry1 = max(0, ry - 10)
    rx2 = min(w, rx + rw + 10)
    ry2 = min(h, ry + rh + 10)
else:
    rx1, ry1, rx2, ry2 = 0, 0, w, h

print(f"Isolated Crate ROI: [{rx1}, {ry1}, {rx2}, {ry2}]")

# STAGE 2: Multi-Scale Adaptive Fruit Detector
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

roi_red1 = (roi_h <= 14) & (roi_s > 70) & (roi_v > 55)
roi_red2 = (roi_h >= 164) & (roi_s > 70) & (roi_v > 55)
roi_orange = (roi_h > 14) & (roi_h <= 28) & (roi_s > 80) & (roi_v > 65)
roi_unripe = (roi_h > 28) & (roi_h <= 45) & (roi_s > 80) & (roi_v > 65)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 45] = 0
roi_mask[roi_s <= 70] = 0

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_clean)
dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

for th_ratio in [0.20, 0.22, 0.25]:
    _, sure_fg = cv2.threshold(dist, th_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    tomatoes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 5:
            continue
        cx, cy = centroids[i]
        r = int(dist[int(cy), int(cx)] * 1.85)
        r = max(9, min(45, r))
        
        gx1 = max(0, int(rx1 + cx - r))
        gy1 = max(0, int(ry1 + cy - r))
        gx2 = min(w, int(rx1 + cx + r))
        gy2 = min(h, int(ry1 + cy + r))
        
        crop = img[gy1:gy2, gx1:gx2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        
        if (mean_h <= 14 or mean_h >= 164):
            stage = "ripe"
        elif 15 <= mean_h <= 27:
            stage = "overripe"
        elif 28 <= mean_h <= 45:
            stage = "unripe"
        else:
            stage = "ripe"
            
        tomatoes.append(stage)
        
    counts = {"ripe": tomatoes.count("ripe"), "unripe": tomatoes.count("unripe"), "overripe": tomatoes.count("overripe"), "spoiled": tomatoes.count("spoiled")}
    print(f"Threshold {th_ratio}: Total {len(tomatoes)} tomatoes detected -> {counts}")
