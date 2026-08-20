import cv2
import numpy as np
import time

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

t0 = time.time()

# ==========================================
# STAGE 1: AUTOMATIC TOMATO CLUSTER ROI LOCALIZATION
# ==========================================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# Core Tomato Color: Red, Orange, Golden (Tomatoes MUST have red/orange presence in a heap)
core_red1 = (h_chan <= 14) & (s_chan > 100) & (v_chan > 70)
core_red2 = (h_chan >= 165) & (s_chan > 100) & (v_chan > 70)
core_orange = (h_chan > 14) & (h_chan <= 25) & (s_chan > 115) & (v_chan > 80)

core_mask = (core_red1 | core_red2 | core_orange).astype(np.uint8) * 255

# Morphological grouping to fuse all tomatoes into the single Crate Cluster ROI
kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
cluster_mask = cv2.dilate(core_mask, kernel_large, iterations=2)
cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

# Find the Main Tomato Cluster Contours
contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Pick the largest tomato cluster contour (The Tomato Crate)
    main_cluster = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_cluster)
    
    # Expand slightly by 15px margin to capture edge tomatoes
    rx1 = max(0, rx - 15)
    ry1 = max(0, ry - 15)
    rx2 = min(w, rx + rw + 15)
    ry2 = min(h, ry + rh + 15)
else:
    rx1, ry1, rx2, ry2 = 0, 0, w, h

print(f"STAGE 1: Isolated Tomato Region (ROI): [{rx1}, {ry1}, {rx2}, {ry2}] (Width: {rx2-rx1}, Height: {ry2-ry1})")

# ==========================================
# STAGE 2: SUB-SECOND INDIVIDUAL TOMATO SEGMENTATION INSIDE ROI ONLY
# ==========================================
# Crop strictly to the Tomato Crate ROI
roi_img = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

# Segment only inside the ROI
roi_red1 = (roi_h <= 13) & (roi_s > 90) & (roi_v > 70)
roi_red2 = (roi_h >= 165) & (roi_s > 90) & (roi_v > 70)
roi_orange = (roi_h > 13) & (roi_h <= 25) & (roi_s > 110) & (roi_v > 75)
roi_unripe = (roi_h > 25) & (roi_h <= 40) & (roi_s > 115) & (roi_v > 80)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 40] = 0 # Strictly NO green lettuce!
roi_mask[roi_s <= 90] = 0 # Strictly NO background!

kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
roi_clean = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
dist_transform = cv2.distanceTransform(roi_clean, cv2.DIST_L2, 5)

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
        r = int(dist_transform[int(cy_roi), int(cx_roi)] * 1.85)
        r = max(11, min(40, r))
        
        # Map back to global full-image coordinates
        global_cx = rx1 + cx_roi
        global_cy = ry1 + cy_roi
        
        gx1 = max(0, int(global_cx - r))
        gy1 = max(0, int(global_cy - r))
        gx2 = min(w, int(global_cx + r))
        gy2 = min(h, int(global_cy + r))
        
        crop = img[gy1:gy2, gx1:gx2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        
        if (mean_h <= 13 or mean_h >= 165):
            stage = "fresh"
            color = (46, 213, 115)
        elif 14 <= mean_h <= 25:
            stage = "overripe"
            color = (255, 159, 67)
        elif 26 <= mean_h <= 40:
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

# Draw ROI bounding box in white/cyan
cv2.rectangle(output_img, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

elapsed_ms = (time.time() - t0) * 1000
print(f"Elapsed Time: {elapsed_ms:.1f} ms (Sub-second / High-speed!)")
print(f"Total True Tomatoes inside Crate ROI: {len(tomatoes)}")
counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for t in tomatoes:
    counts[t["stage"]] += 1
print(f"Counts: {counts}")

cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/two_stage_detected.jpg", output_img)
