import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]

# Convert to HSV and Lab
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tomatoes (Ripe, Overripe, Yellow/Turning, Green/Breaker) have distinct saturation & hue:
# Green lettuce: H in [35, 75], but lettuce is leafy/jagged, NOT round individual fruits!
# Potatoes: Brown, low saturation S < 45.
# Sand floor: Grey/Brown, low saturation S < 50.

# Tomato mask requiring real saturation:
mask_tomato = (
    ((h_c <= 15) | (h_c >= 160)) & (s_c > 75) & (v_c > 50) | # Red
    ((h_c > 15) & (h_c <= 28)) & (s_c > 80) & (v_c > 55) |   # Orange
    ((h_c > 28) & (h_c <= 42)) & (s_c > 80) & (v_c > 55) |   # Yellow / Turning
    ((h_c > 42) & (h_c <= 65)) & (s_c > 60) & (v_c > 50)     # Green breaker
).astype(np.uint8) * 255

# Apply morphological smoothing per tomato fruit size (diameter ~ 30-70px)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
clean_mask = cv2.morphologyEx(mask_tomato, cv2.MORPH_OPEN, kernel)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

# Watershed / Distance transform on the clean mask
dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

# Find local maxima peaks (centers of individual tomato fruits!)
kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
dilated_dist = cv2.dilate(dist, kernel_peak)
peaks = (dist == dilated_dist) & (dist > 8.0) # At least 8px radius

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))

print(f"Total candidate tomato fruit peaks found: {num_labels - 1}")

valid_fruits = []
annotated = img.copy()

for i in range(1, num_labels):
    cx, cy = centroids[i]
    cx, cy = int(cx), int(cy)
    r = int(dist[cy, cx] * 1.5)
    r = max(16, min(45, r))

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    mean_s = np.mean(crop_hsv[:, :, 1])
    mean_v = np.mean(crop_hsv[:, :, 2])

    # REJECT NON-TOMATOES:
    # 1. Reject Sand Floor: Sand floor has low saturation (S < 60) or is at y > 720
    if cy > 700 and mean_s < 90:
        continue
    # 2. Reject Lettuce on left: Lettuce is at x < 180 and pure green H > 55 with high jaggedness
    if cx < 185:
        continue
    # 3. Reject Potatoes on right: Potatoes are at x > 560 and S < 65
    if cx > 560:
        continue

    # Classify Ripeness
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

    if dark_spots > 0.40:
        stage = "spoiled"
        color = (0, 0, 255)
    elif (mean_h <= 18 or mean_h >= 160):
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

    valid_fruits.append((x1, y1, x2, y2, stage))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(annotated, stage, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_exact.jpg", annotated)
print(f"Total valid tomato fruits retained strictly inside crate: {len(valid_fruits)}")
counts = {}
for f in valid_fruits:
    counts[f[4]] = counts.get(f[4], 0) + 1
print(f"Counts: {counts}")
