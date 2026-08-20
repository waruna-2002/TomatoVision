import cv2
import numpy as np

img_path = r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

core_red1 = (h_chan <= 14) & (s_chan > 70) & (v_chan > 60)
core_red2 = (h_chan >= 164) & (s_chan > 70) & (v_chan > 60)
core_orange = (h_chan > 14) & (h_chan <= 28) & (s_chan > 80) & (v_chan > 70)
core_unripe = (h_chan > 28) & (h_chan <= 48) & (s_chan > 70) & (v_chan > 70)

core_mask = (core_red1 | core_red2 | core_orange | core_unripe).astype(np.uint8) * 255
core_mask[h_chan > 50] = 0
core_mask[s_chan <= 60] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)
dist_transform = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

tomatoes = []
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < 20:
        continue
    cx, cy = centroids[i]
    r = int(dist_transform[int(cy), int(cx)] * 1.8)
    r = max(20, min(180, r))
    tomatoes.append((cx, cy, r))

print(f"Single image detections: {len(tomatoes)}")
