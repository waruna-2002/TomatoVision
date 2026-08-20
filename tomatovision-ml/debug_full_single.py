import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

red_orange_core = (
    ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
    ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Contours in single tomato photo:")
for c in contours:
    area = cv2.contourArea(c)
    bx, by, bw, bh = cv2.boundingRect(c)
    print(f"  Area={area:.0f}, BBox=[{bx}, {by}, {bw}, {bh}]")
