import cv2
import numpy as np
import time

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

t0 = time.time()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# 1. Detect ONLY core ripe/orange tomato pixels (Exclude all green leaves & dark cabbage)
core_red1 = (h_chan <= 13) & (s_chan > 110) & (v_chan > 80)
core_red2 = (h_chan >= 166) & (s_chan > 110) & (v_chan > 80)
core_orange = (h_chan > 13) & (h_chan <= 24) & (s_chan > 120) & (v_chan > 85)

core_mask = (core_red1 | core_red2 | core_orange).astype(np.uint8) * 255

# Clean single stray noise
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_small)

# Connect dense tomato cluster
kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
cluster_mask = cv2.dilate(core_mask, kernel_group, iterations=2)

contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Find the largest cluster by area with highest tomato density
    main_cluster = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cluster)
    rx1 = max(0, rx - 8)
    ry1 = max(0, ry - 8)
    rx2 = min(w, rx + rw + 8)
    ry2 = min(h, ry + rh + 8)
else:
    rx1, ry1, rx2, ry2 = 0, 0, w, h

print(f"STAGE 1 Tomato Crate Isolated: X=[{rx1}, {rx2}], Y=[{ry1}, {ry2}]")

# ==========================================
# STAGE 2: Individual Tomato Detection inside Crate
# ==========================================
roi_img = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

# Pigment inside tomato crate only
roi_red1 = (roi_h <= 13) & (roi_s > 95) & (roi_v > 75)
roi_red2 = (roi_h >= 165) & (roi_s > 95) & (roi_v > 75)
roi_orange = (roi_h > 13) & (roi_h <= 24) & (roi_s > 115) & (roi_v > 80)
roi_unripe = (roi_h > 24) & (roi_h <= 36) & (roi_s > 120) & (roi_v > 85) # strictly breaker yellow-green

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 36] = 0 # 100% EXCLUDE ANY GREEN LEAF
roi_mask[roi_s <= 95] = 0 # 100% EXCLUDE SAND/WOOD/POTATO

dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)

tomatoes = []
output_img = img.copy()

if dist_transform.max() > 0:
    _, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 8:
            continue
        cx_roi, cy_roi = centroids[i]
        r = int(dist_transform[int(cy_roi), int(cx_roi)] * 1.8)
        r = max(11, min(38, r))
        
        gx1 = max(0, int(rx1 + cx_roi - r))
        gy1 = max(0, int(ry1 + cy_roi - r))
        gx2 = min(w, int(rx1 + cx_roi + r))
        gy2 = min(h, int(ry1 + cy_roi + r))
        
        crop = img[gy1:gy2, gx1:gx2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        
        if (mean_h <= 13 or mean_h >= 165):
            stage = "fresh"
            color = (46, 213, 115)
        elif 14 <= mean_h <= 24:
            stage = "overripe"
            color = (255, 159, 67)
        elif 25 <= mean_h <= 36:
            stage = "unripe"
            color = (0, 210, 211)
        else:
            stage = "fresh"
            color = (46, 213, 115)
            
        tomatoes.append({
            "stage": stage,
            "box": [gx1, gy1, gx2, gy2]
        })
        cv2.rectangle(output_img, (gx1, gy1), (gx2, gy2), color, 2)

elapsed_ms = (time.time() - t0) * 1000
print(f"Speed: {elapsed_ms:.1f} milliseconds (Ultra-Fast!)")
print(f"Total True Tomatoes in Crate: {len(tomatoes)}")
counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for t in tomatoes:
    counts[t["stage"]] += 1
print(f"Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/clean_two_stage.jpg", output_img)
