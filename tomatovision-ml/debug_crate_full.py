import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
h, w = img.shape[:2]
img_area = float(w * h)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# In crate of tomatoes, why did S > 110 only match a small portion?
# Because the crate has ripe red tomatoes, orange tomatoes, AND green unripe tomatoes with S around 80-130!
# When tomato_seeds uses S > 80 for red/orange, what happens?
tomato_seeds = (
    ((h_chan <= 14) | (h_chan >= 165)) & (s_chan > 75) & (v_chan > 50) |
    ((h_chan > 14) & (h_chan <= 28)) & (s_chan > 80) & (v_chan > 55) |
    ((h_chan > 28) & (h_chan <= 60)) & (s_chan > 70) & (v_chan > 50)
).astype(np.uint8) * 255

kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
dilated_seeds = cv2.dilate(tomato_seeds, kernel_heap, iterations=2)
heap_contours, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total heap contours: {len(heap_contours)}")
for i, c in enumerate(heap_contours):
    area = cv2.contourArea(c)
    hx, hy, hw, hh = cv2.boundingRect(c)
    print(f"  Contour {i}: Area={area:.0f}, BBox=[x={hx}, y={hy}, w={hw}, h={hh}]")
