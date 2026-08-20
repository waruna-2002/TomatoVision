import cv2
import numpy as np

# Load the single green tomato photo
img_full = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102573801.png")
cam = img_full[58:518, 412:612]
h, w = cam.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# 1. Reject the top/bottom 10% frame edges if they are straight horizontal background bands
# 2. Extract fruit color masks:
# Green/Breaker tomato: H in [25, 60], S > 30, V > 50
# Red tomato: H <= 13 or H >= 165, S > 70, V > 50
# Orange/Overripe: H in [14, 25], S > 75, V > 50
# Yellow: H in [26, 35], S > 70, V > 50

mask_tomato = (
    ((h_c <= 13) | (h_c >= 165)) & (s_c > 70) & (v_c > 50) |
    ((h_c > 13) & (h_c <= 25)) & (s_c > 75) & (v_c > 50) |
    ((h_c > 25) & (h_c <= 35)) & (s_c > 65) & (v_c > 50) |
    ((h_c > 35) & (h_c <= 60)) & (s_c > 30) & (v_c > 50)
).astype(np.uint8) * 255

# Clean morphology
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
closed = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel_close)
clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours found in single tomato scene: {len(contours)}")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, cw, ch = cv2.boundingRect(c)
    ar = float(cw)/ch if ch > 0 else 0
    perimeter = cv2.arcLength(c, True)
    circ = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    print(f"Contour {i}: Area={area:.0f}, BBox=[{x},{y},{cw},{ch}], AR={ar:.2f}, Circ={circ:.2f}, Solidity={solidity:.2f}")
