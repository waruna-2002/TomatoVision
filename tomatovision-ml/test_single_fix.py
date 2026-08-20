import cv2
import numpy as np

img4 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :] # Single green tomato

h, w = img4.shape[:2]
hsv = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

red_orange_core = (
    ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
    ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_c = []
if contours:
    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = float(bw) / bh if bh > 0 else 0
        # Reject background table strips at the border (touching top, bottom, or wide)
        is_table = (by <= 10 and bw > 0.40 * w) or (by + bh >= h - 10 and bw > 0.40 * w) or (ar > 3.0)
        if area > 3500 and not is_table:
            valid_c.append(c)

print(f"Valid heap contours in single tomato photo: {len(valid_c)}")
