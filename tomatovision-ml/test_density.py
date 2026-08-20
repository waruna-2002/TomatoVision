import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# Pure Tomato Pigment (Ripe Red, Overripe Orange, Yellow Turning)
tomato_core = (
    ((h_c <= 16) | (h_c >= 165)) & (s_c > 115) & (v_c > 55) |
    ((h_c > 16) & (h_c <= 28)) & (s_c > 120) & (v_c > 60) |
    ((h_c > 28) & (h_c <= 40)) & (s_c > 120) & (v_c > 60)
).astype(np.uint8) * 255

# ONLY small morphological close (7x7) to connect pixels WITHIN the same tomato fruit!
clean_seeds = cv2.morphologyEx(tomato_core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

# Find connected tomato fruit components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_seeds)

fruit_components = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 120: # A real tomato spot
        cx, cy = centroids[i]
        fruit_components.append((cx, cy, area))

print(f"Total individual tomato fruit spots found: {len(fruit_components)}")

# Find the spatial cluster of tomato fruits (Tomato Crate / Heap Bounding Box)
all_cx = np.array([f[0] for f in fruit_components])
all_cy = np.array([f[1] for f in fruit_components])

# Filter outliers (keep the main dense cluster)
median_x = np.median(all_cx)
median_y = np.median(all_cy)

# Keep points within 2 standard deviations of the median
std_x = np.std(all_cx)
std_y = np.std(all_cy)

in_cluster = (np.abs(all_cx - median_x) < 1.6 * std_x) & (np.abs(all_cy - median_y) < 1.6 * std_y)
cluster_cx = all_cx[in_cluster]
cluster_cy = all_cy[in_cluster]

hx1 = max(0, int(np.min(cluster_cx) - 30))
hy1 = max(0, int(np.min(cluster_cy) - 30))
hx2 = min(w, int(np.max(cluster_cx) + 30))
hy2 = min(h, int(np.max(cluster_cy) + 30))

print(f"ISOLATED TOMATO TRAY BOUNDING BOX: [{hx1}, {hy1}, {hx2}, {hy2}] (Width={hx2-hx1}, Height={hy2-hy1})")

# Segment inside the ISOLATED TOMATO TRAY ONLY!
roi = img[hy1:hy2, hx1:hx2]
roi_hsv = hsv[hy1:hy2, hx1:hx2]
roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

roi_mask = (
    ((roi_h <= 16) | (roi_h >= 162)) & (roi_s > 65) & (roi_v > 45) |
    ((roi_h > 16) & (roi_h <= 28)) & (roi_s > 70) & (roi_v > 50) |
    ((roi_h > 28) & (roi_h <= 42)) & (roi_s > 70) & (roi_v > 50) |
    ((roi_h > 42) & (roi_h <= 65)) & (roi_s > 55) & (roi_v > 45)
).astype(np.uint8) * 255

clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)
kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
dilated_dist = cv2.dilate(dist, kernel_peak)
local_max = (dist == dilated_dist) & (dist > 5.0) & (dist > 0.08 * dist.max())

num_labels_roi, _, _, centroids_roi = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
raw_centers = []
for i in range(1, num_labels_roi):
    cx, cy = centroids_roi[i]
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
cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (255, 0, 0), 4) # Big Blue Box around Tomato Tray

counts = {}
for cx, cy, d_val in suppressed:
    r = int(d_val * 1.55)
    r = max(14, min(38, r))

    gx1 = max(0, int(hx1 + cx - r))
    gy1 = max(0, int(hy1 + cy - r))
    gx2 = min(w, int(hx1 + cx + r))
    gy2 = min(h, int(hy1 + cy + r))

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

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_density.jpg", annotated)
print(f"Final Counts: {counts}")
