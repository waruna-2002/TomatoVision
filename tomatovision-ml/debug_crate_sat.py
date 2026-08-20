import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

print(f"Crate image shape: {w}x{h}")
for s_val in [70, 85, 100, 115]:
    seeds = (
        ((h_chan <= 14) | (h_chan >= 162)) & (s_chan > s_val) & (v_chan > 50) |
        ((h_chan > 14) & (h_chan <= 28)) & (s_chan > s_val) & (v_chan > 55)
    ).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated = cv2.dilate(seeds, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n--- S > {s_val} ---")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, cw, ch = cv2.boundingRect(c)
            print(f"  Contour {i}: Area={area:.0f}, BBox=[x={x}, y={y}, w={cw}, h={ch}]")
