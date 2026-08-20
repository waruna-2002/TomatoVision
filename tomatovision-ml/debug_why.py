import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# In api_server.py:
tomato_seeds = (
    ((h_chan <= 13) | (h_chan >= 165)) & (s_chan > 115) & (v_chan > 55) |
    ((h_chan > 13) & (h_chan <= 25)) & (s_chan > 125) & (v_chan > 60)
).astype(np.uint8) * 255

kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
dilated_seeds = cv2.dilate(tomato_seeds, kernel_heap, iterations=2)
heap_contours, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total dilated heap contours found: {len(heap_contours)}")
for i, c in enumerate(heap_contours):
    c_area = cv2.contourArea(c)
    hx, hy, hw, hh = cv2.boundingRect(c)
    print(f"  Contour {i}: Area={c_area:.0f} ({(c_area/img_area)*100:.1f}%), BBox=[x={hx}, y={hy}, w={hw}, h={hh}]")
