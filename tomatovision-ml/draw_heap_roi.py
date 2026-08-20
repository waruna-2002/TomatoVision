import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

red_orange_seeds = (
    ((h_chan <= 14) | (h_chan >= 160)) & (s_chan > 70) & (v_chan > 50) |
    ((h_chan > 14) & (h_chan <= 28)) & (s_chan > 75) & (v_chan > 55)
).astype(np.uint8) * 255

kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
dilated_seeds = cv2.dilate(red_orange_seeds, kernel_heap, iterations=2)
heap_contours, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_heaps = []
for c in heap_contours:
    c_area = cv2.contourArea(c)
    if c_area < 2500:
        continue
    hx, hy, hw, hh = cv2.boundingRect(c)
    ar = float(hw) / hh if hh > 0 else 0
    if ar <= 3.0 and not (hw > 0.70 * w and (hy <= 5 or hh < 0.12 * h)):
        valid_heaps.append((c, hx, hy, hw, hh, c_area))

main_heap = max(valid_heaps, key=lambda x: x[5])
_, hx, hy, hw, hh, _ = main_heap

print(f"Image dimensions: width={w}, height={h}")
print(f"Detected Heap ROI: x={hx}, y={hy}, width={hw}, height={hh}")

annotated = img.copy()
# Draw heap ROI in thick blue
cv2.rectangle(annotated, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 4)

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_heap.jpg", annotated)
print("Saved test_annotated_heap.jpg")
