import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png")
cam = img[58:518, 412:612]
cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_images\cam_green.jpg", cam)
print("Saved cam_green.jpg, shape:", cam.shape)

hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# In this photo: Paper is white/grey (S < 35), wooden table is brown on top (y < 80)
# The green tomato is in the center
# Green tomato color: Hue 28 - 48, Saturation > 35, Value > 70
mask_green_tomato = (h_c >= 25) & (h_c <= 60) & (s_c > 35) & (v_c > 70) & (np.arange(cam.shape[0])[:, None] > 60)
mask_u8 = mask_green_tomato.astype(np.uint8) * 255

clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(contours)}")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, cw, ch = cv2.boundingRect(c)
    ar = float(cw)/ch if ch > 0 else 0
    perimeter = cv2.arcLength(c, True)
    circ = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    print(f"Contour {i}: Area={area:.0f}, BBox=[{x},{y},{cw},{ch}], AR={ar:.2f}, Circ={circ:.2f}")
