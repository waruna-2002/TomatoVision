import cv2
import numpy as np

img_bgr = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img_bgr.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask_red = ((h_c <= 15) | (h_c >= 160)) & (s_c > 75) & (v_c > 45)
mask_orange = (h_c > 15) & (h_c <= 28) & (s_c > 80) & (v_c > 50)
mask_yellow = (h_c > 28) & (h_c <= 42) & (s_c > 80) & (v_c > 50)
mask_breaker = (h_c > 42) & (h_c <= 65) & (s_c > 70) & (v_c > 50)

tomato_core = (mask_red | mask_orange | mask_yellow).astype(np.uint8) * 255
print("Nonzero in tomato_core:", cv2.countNonZero(tomato_core))

kernel_group = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
dilated_core = cv2.morphologyEx(tomato_core, cv2.MORPH_CLOSE, kernel_group)
dilated_core = cv2.dilate(dilated_core, kernel_group, iterations=1)

clusters, _ = cv2.findContours(dilated_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Total clusters found:", len(clusters))

valid_clusters = []
for i, c in enumerate(clusters):
    area = cv2.contourArea(c)
    bx, by, bw, bh = cv2.boundingRect(c)
    ar = float(bw) / bh if bh > 0 else 0
    cond1 = area > max(3000, 0.015 * img_area)
    cond2 = ar < 3.2
    cond3 = not (bw > 0.80 * w and by <= 5 and bh < 0.12 * h)
    print(f"Cluster {i}: Area={area:.0f} (thresh={max(3000, 0.015*img_area):.0f}), BBox=[{bx},{by},{bw},{bh}], AR={ar:.2f}, cond1={cond1}, cond2={cond2}, cond3={cond3}")
    if cond1 and cond2 and cond3:
        valid_clusters.append((c, bx, by, bw, bh, area))

print("Valid clusters:", len(valid_clusters))
