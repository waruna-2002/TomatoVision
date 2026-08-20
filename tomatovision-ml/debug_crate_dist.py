import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rx1, ry1, rx2, ry2 = 16, 1, 457, 672
roi = img[ry1:ry2, rx1:rx2]
roi_hsv = hsv[ry1:ry2, rx1:rx2]
roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

roi_red1 = (roi_h <= 14) & (roi_s > 70) & (roi_v > 45)
roi_red2 = (roi_h >= 162) & (roi_s > 70) & (roi_v > 45)
roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 75) & (roi_v > 50)
roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 75) & (roi_v > 50)
roi_unripe = (roi_h > 36) & (roi_h <= 60) & (roi_s > 60) & (roi_v > 50)

roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)
print(f"Dist max in crate ROI: {dist.max()}")

kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
dilated = cv2.dilate(dist, kernel_peak)
local_max = (dist == dilated) & (dist > 4.0) & (dist > 0.08 * dist.max())

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
print(f"Total raw peaks found in crate ROI: {num_labels - 1}")
