import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")
h, w = img.shape[:2]
img_area = float(w * h)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask_tomato = (
    ((h_chan <= 15) | (h_chan >= 162)) & (s_chan > 50) & (v_chan > 40) |
    ((h_chan > 15) & (h_chan <= 28)) & (s_chan > 60) & (v_chan > 45) |
    ((h_chan > 28) & (h_chan <= 38)) & (s_chan > 60) & (v_chan > 45) |
    ((h_chan > 38) & (h_chan <= 58)) & (s_chan > 50) & (v_chan > 40)
).astype(np.uint8) * 255

mask_tomato[h_chan > 58] = 0
mask_tomato[(h_chan >= 120) & (h_chan <= 165)] = 0
mask_tomato[s_chan <= 35] = 0

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_mask = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel_close)
clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")

for i, c in enumerate(contours):
    c_area = cv2.contourArea(c)
    if c_area < 500:
        continue
    x, y, cw, ch = cv2.boundingRect(c)
    ar = float(cw) / ch if ch > 0 else 0
    perimeter = cv2.arcLength(c, True)
    circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
    print(f"Contour {i}: Area={c_area:.0f} ({(c_area/img_area)*100:.2f}%), BBox=[{x},{y},{cw},{ch}], AR={ar:.2f}, Circ={circ:.2f}")
