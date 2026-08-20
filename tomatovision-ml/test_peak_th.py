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
core_red1 = (h_chan <= 15) & (s_chan > 50) & (v_chan > 40)
core_red2 = (h_chan >= 162) & (s_chan > 50) & (v_chan > 40)
core_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 60) & (v_chan > 50)
core_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 60) & (v_chan > 50)
core_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 50) & (v_chan > 45)

core_mask = (core_red1 | core_red2 | core_orange | core_yellow | core_unripe).astype(np.uint8) * 255
# Exclude outside leaves (H > 55) and cabbage (H in 120-165)
core_mask[h_chan > 55] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 45] = 0

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

print(f"Crate ROI: [{rx1}, {ry1}, {rx2}, {ry2}] (W={rx2-rx1}, H={ry2-ry1})")

# 2. Local Peak / Distance Transform Detection inside Crate
roi_img = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

roi_red1 = (roi_h <= 15) & (roi_s > 45) & (roi_v > 40)
roi_red2 = (roi_h >= 162) & (roi_s > 45) & (roi_v > 40)
roi_orange = (roi_h > 15) & (roi_h <= 28) & (roi_s > 50) & (roi_v > 45)
roi_yellow = (roi_h > 28) & (roi_h <= 38) & (roi_s > 50) & (roi_v > 45)
roi_unripe = (roi_h > 38) & (roi_h <= 55) & (roi_s > 45) & (roi_v > 40)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 55] = 0
roi_mask[roi_s <= 45] = 0

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_clean)
dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

# Use lower peak threshold for complete fruit coverage (0.12 - 0.16)
for peak_th in [0.12, 0.15, 0.18]:
    _, sure_fg = cv2.threshold(dist, peak_th * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    tomatoes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        cx, cy = centroids[i]
        r = int(dist[int(cy), int(cx)] * 1.7)
        r = max(8, min(42, r))
        
        gx1 = max(0, int(rx1 + cx - r))
        gy1 = max(0, int(ry1 + cy - r))
        gx2 = min(w, int(rx1 + cx + r))
        gy2 = min(h, int(ry1 + cy + r))
        
        crop = img[gy1:gy2, gx1:gx2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        
        # Check spoilage / dark rot lesion
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_spots = np.sum(gray_crop < 40) / float(gray_crop.size) if gray_crop.size > 0 else 0
        
        if dark_spots > 0.28:
            stage = "spoiled"
        elif (mean_h <= 13 or mean_h >= 164):
            stage = "ripe"
        elif 14 <= mean_h <= 26:
            stage = "overripe"
        elif 27 <= mean_h <= 55:
            stage = "unripe"
        else:
            stage = "ripe"
            
        tomatoes.append(stage)
        
    counts = {
        "ripe": tomatoes.count("ripe"),
        "unripe": tomatoes.count("unripe"),
        "overripe": tomatoes.count("overripe"),
        "spoiled": tomatoes.count("spoiled")
    }
    print(f"Peak TH {peak_th}: Total {len(tomatoes)} tomatoes detected -> {counts}")
