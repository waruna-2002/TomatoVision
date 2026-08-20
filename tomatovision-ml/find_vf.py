import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png")
h, w = img.shape[:2]
print(f"Full Screenshot Shape: ({h}, {w})")

# Let us find the non-dark area in the upper middle
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# In the app, background is dark #0A0E1A (value < 40)
# Find bounding box of bright region where value > 40
mask_bright = (gray > 45).astype(np.uint8) * 255
# Only look in y: 50..550
roi = mask_bright[50:550, :]
contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    x, y, cw, ch = cv2.boundingRect(c)
    if cw > 100 and ch > 100:
        print(f"Camera Viewport {i}: [{x}, {y+50}, {cw}, {ch}]")
