import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")
h, w = img.shape[:2]
img_area = float(w * h)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask_tomato = (
    ((h_chan <= 15) | (h_chan >= 162)) & (s_chan > 45) & (v_chan > 40) |
    ((h_chan > 15) & (h_chan <= 28)) & (s_chan > 55) & (v_chan > 45) |
    ((h_chan > 28) & (h_chan <= 38)) & (s_chan > 55) & (v_chan > 45) |
    ((h_chan > 38) & (h_chan <= 58)) & (s_chan > 45) & (v_chan > 40)
).astype(np.uint8) * 255

mask_tomato[h_chan > 58] = 0
mask_tomato[(h_chan >= 120) & (h_chan <= 165)] = 0
mask_tomato[s_chan <= 35] = 0

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
closed_mask = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel_close)
clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segmented = []

for c in contours:
    c_area = cv2.contourArea(c)
    if c_area < (0.010 * img_area):
        continue
    x, y, cw, ch = cv2.boundingRect(c)
    ar = float(cw) / ch if ch > 0 else 0
    perimeter = cv2.arcLength(c, True)
    circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0

    # Reject elongated table strips / borders
    if 0.50 <= ar <= 2.2 and circ >= 0.30:
        if not (cw > 0.60 * w and ch < 0.20 * h):
            crop = img[y:y+ch, x:x+cw]
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(crop_hsv[:, :, 0])
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0

            if dark_spots > 0.30:
                stage = "spoiled"
            elif (mean_h <= 13 or mean_h >= 164):
                stage = "ripe"
            elif 14 <= mean_h <= 26:
                stage = "overripe"
            elif 27 <= mean_h <= 58:
                stage = "unripe"
            else:
                stage = "ripe"

            segmented.append({
                "class_name": stage,
                "box": [x, y, x + cw, y + ch]
            })

print(f"Result on Single Green Tomato: {len(segmented)} fruit detected -> {segmented}")
