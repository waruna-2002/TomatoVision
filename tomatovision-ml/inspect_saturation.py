import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# Why did the dirt sack trigger red_orange_seeds?
# Because dirt sack has low saturation (S ~ 50-80).
# Real ripe/orange tomatoes have VIBRANT, HIGH SATURATION (S > 120)!

print("Let us test different Saturation thresholds for real tomatoes:")
for s_thresh in [80, 100, 120, 140]:
    seeds = (
        ((h_chan <= 13) | (h_chan >= 165)) & (s_chan > s_thresh) & (v_chan > 60) |
        ((h_chan > 13) & (h_chan <= 25)) & (s_chan > (s_thresh + 10)) & (v_chan > 65)
    ).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated = cv2.dilate(seeds, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n--- Saturation Thresh > {s_thresh} ---")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, cw, ch = cv2.boundingRect(c)
            print(f"  Contour {i}: Area={area:.0f}, BBox=[x={x}, y={y}, w={cw}, h={ch}]")
