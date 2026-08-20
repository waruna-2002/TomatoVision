import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787106806236.png")
h_all, w_all = img.shape[:2]
# Extract camera viewport
viewport = img[88:544, 403:597]

h, w = viewport.shape[:2]
hsv = cv2.cvtColor(viewport, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

red_orange_core = (
    ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
    ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Heap contours in single green photo:")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    bx, by, bw, bh = cv2.boundingRect(c)
    ar = float(bw) / bh if bh > 0 else 0
    is_table_edge = (by <= int(0.12 * h) and bh < int(0.20 * h)) or (ar > 3.0) or (bw > 0.80 * w and by <= 5)
    print(f"  Contour {i}: Area={area:.0f}, BBox=[{bx},{by},{bw},{bh}], AR={ar:.2f}, is_table_edge={is_table_edge}")

# What did api_server.py return on viewport?
import api_server
res = api_server.detect_tomatoes_two_stage(viewport)
print("api_server.py returned:", res)
