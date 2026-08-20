import cv2
import numpy as np

img_path = r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

mask_red1 = (h_chan <= 18) & (s_chan > 50) & (v_chan > 50)
mask_red2 = (h_chan >= 162) & (s_chan > 50) & (v_chan > 50)
mask_orange = (h_chan > 18) & (h_chan <= 38) & (s_chan > 60) & (v_chan > 60)
mask_green = (h_chan > 38) & (h_chan <= 80) & (s_chan > 50) & (v_chan > 50)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_green).astype(np.uint8) * 255
cabbage_mask = (h_chan >= 125) & (h_chan <= 160) & (s_chan > 40)
tomato_mask[cabbage_mask] = 0

gray_mask = (s_chan < 35) | (v_chan < 35)
tomato_mask[gray_mask] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
detected = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 30:
        continue
    cx, cy = centroids[i]
    r = int(dist_transform[int(cy), int(cx)] * 2.0)
    r = max(20, min(200, r))
    
    x1 = max(0, int(cx - r))
    y1 = max(0, int(cy - r))
    x2 = min(w, int(cx + r))
    y2 = min(h, int(cy + r))
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv_crop[:, :, 0])
    mean_s = np.mean(hsv_crop[:, :, 1])
    
    if (mean_h <= 14 or mean_h >= 165) and mean_s > 60:
        stage = "fresh"
    elif 15 <= mean_h <= 36:
        stage = "overripe"
    elif 37 <= mean_h <= 85:
        stage = "unripe"
    else:
        stage = "fresh"
    detected.append(stage)

print(f"Test Tomatoes detected: {len(detected)} -> {detected}")
