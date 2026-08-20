import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Loaded Market Image: {w}x{h}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# 1. Strict Tomato Anchor Pigments (Red & Vibrant Orange)
# Ground, wood, green chilies, cucumbers, beetroots have 0 overlap with this!
anchor_red1 = (h_chan <= 13) & (s_chan > 70) & (v_chan > 55)
anchor_red2 = (h_chan >= 165) & (s_chan > 70) & (v_chan > 55)
anchor_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 80) & (v_chan > 65)

anchor_mask = (anchor_red1 | anchor_red2 | anchor_orange).astype(np.uint8) * 255

# Group the tomato seeds into the Tomato Heap ROI
kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
heap_clustered = cv2.dilate(anchor_mask, kernel_heap, iterations=2)

contours, _ = cv2.findContours(heap_clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No tomato cluster found!")
    exit(0)

main_cluster = max(contours, key=cv2.contourArea)
rx, ry, rw, rh = cv2.boundingRect(main_cluster)
print(f"Isolated Tomato Heap BBox: X=[{rx}, {rx+rw}], Y=[{ry}, {ry+rh}] (Width={rw}, Height={rh})")

# Expand bounding box slightly (15px) to include edge tomatoes in heap
rx1 = max(0, rx - 12)
ry1 = max(0, ry - 12)
rx2 = min(w, rx + rw + 12)
ry2 = min(h, ry + rh + 12)

# Extract Heap ROI
roi_img = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h = roi_hsv[:, :, 0]
roi_s = roi_hsv[:, :, 1]
roi_v = roi_hsv[:, :, 2]

# Mask for tomatoes inside heap (includes red, orange, yellow, and unripe breaker inside the heap)
roi_red1 = (roi_h <= 14) & (roi_s > 60) & (roi_v > 50)
roi_red2 = (roi_h >= 164) & (roi_s > 60) & (roi_v > 50)
roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 70) & (roi_v > 55)
roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 70) & (roi_v > 55)
roi_unripe = (roi_h > 36) & (roi_h <= 50) & (roi_s > 70) & (roi_v > 55)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
roi_mask[roi_h > 50] = 0
roi_mask[roi_s <= 55] = 0

clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
dilated = cv2.dilate(dist, kernel_peak)
local_max = (dist == dilated) & (dist > 4.5) & (dist > 0.08 * dist.max())

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))

raw_centers = []
for i in range(1, num_labels):
    cx, cy = centroids[i]
    d_val = dist[int(cy), int(cx)]
    raw_centers.append((cx, cy, d_val))

raw_centers.sort(key=lambda x: x[2], reverse=True)

suppressed = []
min_dist_sq = 15.0 ** 2

for c in raw_centers:
    cx, cy, d_val = c
    too_close = False
    for s in suppressed:
        if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
            too_close = True
            break
    if not too_close:
        suppressed.append((cx, cy, d_val))

tomatoes = []
output = img.copy()

# Draw the isolated heap boundary
cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (255, 255, 0), 3)

for cx, cy, d_val in suppressed:
    r = int(d_val * 1.65)
    r = max(10, min(40, r))
    
    gx1 = max(0, int(rx1 + cx - r))
    gy1 = max(0, int(ry1 + cy - r))
    gx2 = min(w, int(rx1 + cx + r))
    gy2 = min(h, int(ry1 + cy + r))
    
    crop = img[gy1:gy2, gx1:gx2]
    if crop.size == 0:
        continue
        
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0
    
    if dark_spots > 0.28:
        stage = "spoiled"
        color = (0, 0, 255)
    elif (mean_h <= 13 or mean_h >= 164):
        stage = "ripe"
        color = (0, 255, 0)
    elif 14 <= mean_h <= 26:
        stage = "overripe"
        color = (0, 165, 255)
    elif 27 <= mean_h <= 55:
        stage = "unripe"
        color = (255, 255, 0)
    else:
        stage = "ripe"
        color = (0, 255, 0)
        
    tomatoes.append(stage)
    cv2.rectangle(output, (gx1, gy1), (gx2, gy2), color, 2)

counts = {
    "ripe": tomatoes.count("ripe"),
    "unripe": tomatoes.count("unripe"),
    "overripe": tomatoes.count("overripe"),
    "spoiled": tomatoes.count("spoiled")
}

print(f"\nSUCCESS! Total Tomatoes in Heap Detected: {len(tomatoes)}")
print(f"Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/market_isolated_heap.jpg", output)
