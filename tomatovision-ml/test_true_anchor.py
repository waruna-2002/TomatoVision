import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# 1. PURE TOMATO PIGMENT (S > 130 completely excludes sand S=72, potatoes S=37, and wood S=50)
# Tomato Hues: Red (H <= 16 or H >= 165), Orange (16 < H <= 27), Yellow (27 < H <= 37)
tomato_core = (
    ((h_c <= 16) | (h_c >= 165)) & (s_c > 130) & (v_c > 60) |
    ((h_c > 16) & (h_c <= 27)) & (s_c > 135) & (v_c > 65) |
    ((h_c > 27) & (h_c <= 37)) & (s_c > 135) & (v_c > 65)
).astype(np.uint8) * 255

kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
dilated_core = cv2.dilate(tomato_core, kernel_group, iterations=2)
contours, _ = cv2.findContours(dilated_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours found with S > 130: {len(contours)}")
valid_rois = []
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 10000:
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = float(bw) / bh if bh > 0 else 0
        if ar <= 2.5 and not (bw > 0.75 * w and by <= 5 and bh < 0.12 * h):
            valid_rois.append((bx, by, bw, bh, area))
            print(f"  Valid Tomato Heap ROI: BBox=[x={bx}, y={by}, w={bw}, h={bh}], Area={area:.0f}")

# Now segment inside the Tomato Heap ROI ONLY!
main_heap = max(valid_rois, key=lambda x: x[4])
bx, by, bw, bh, _ = main_heap

# Crop to Tomato Tray!
rx1, ry1 = max(0, bx), max(0, by)
rx2, ry2 = min(w, bx + bw), min(h, by + bh)

roi = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

# Inside the Tray, segment ALL tomatoes (including breaker green ones)
roi_mask = (
    ((roi_h <= 16) | (roi_h >= 162)) & (roi_s > 70) & (roi_v > 50) |
    ((roi_h > 16) & (roi_h <= 28)) & (roi_s > 75) & (roi_v > 50) |
    ((roi_h > 28) & (roi_h <= 42)) & (roi_s > 75) & (roi_v > 50) |
    ((roi_h > 42) & (roi_h <= 65)) & (roi_s > 60) & (roi_v > 50)
).astype(np.uint8) * 255

clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)
kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
dilated_dist = cv2.dilate(dist, kernel_peak)
local_max = (dist == dilated_dist) & (dist > 5.0) & (dist > 0.08 * dist.max())

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
raw_centers = []
for i in range(1, num_labels):
    cx, cy = centroids[i]
    d_val = dist[int(cy), int(cx)]
    raw_centers.append((cx, cy, d_val))

raw_centers.sort(key=lambda item: item[2], reverse=True)

suppressed = []
min_dist_sq = 18.0 ** 2
for pt in raw_centers:
    cx, cy, d_val = pt
    too_close = False
    for s in suppressed:
        if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
            too_close = True
            break
    if not too_close:
        suppressed.append((cx, cy, d_val))

print(f"Total individual tomato fruits extracted: {len(suppressed)}")

annotated = img.copy()
cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3) # Blue box for Tomato Tray

counts = {}
for cx, cy, d_val in suppressed:
    r = int(d_val * 1.55)
    r = max(14, min(38, r))

    gx1 = max(0, int(rx1 + cx - r))
    gy1 = max(0, int(ry1 + cy - r))
    gx2 = min(w, int(rx1 + cx + r))
    gy2 = min(h, int(ry1 + cy + r))

    crop = img[gy1:gy2, gx1:gx2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    if (mean_h <= 18 or mean_h >= 160):
        stage = "ripe"
        color = (0, 255, 0)
    elif 19 <= mean_h <= 28:
        stage = "overripe"
        color = (0, 165, 255)
    elif 29 <= mean_h <= 65:
        stage = "unripe"
        color = (255, 255, 0)
    else:
        stage = "ripe"
        color = (0, 255, 0)

    counts[stage] = counts.get(stage, 0) + 1
    cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 2)
    cv2.putText(annotated, stage, (gx1, gy1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_perfect.jpg", annotated)
print(f"Final Counts: {counts}")
