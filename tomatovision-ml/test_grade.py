import cv2
import numpy as np

def grade_hue(mean_h, dark_spots):
    if dark_spots > 0.28:
        return "spoiled"
    elif (mean_h <= 18 or mean_h >= 160):
        return "ripe"
    elif 19 <= mean_h <= 28:
        return "overripe"
    elif 29 <= mean_h <= 65:
        return "unripe"
    else:
        return "ripe"

# Test on test_unified detections
from test_unified import detect_tomatoes_unified
img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
res = detect_tomatoes_unified(img_m)
reclassified = []
for r in res:
    x1, y1, x2, y2 = map(int, r["box"])
    crop = img_m[y1:y2, x1:x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(crop_hsv[:, :, 0])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_spots = np.sum(gray < 35) / float(gray.size) if gray.size > 0 else 0
    reclassified.append(grade_hue(mean_h, dark_spots))

print(f"Total: {len(reclassified)}")
print(f"Ripe={reclassified.count('ripe')}, Overripe={reclassified.count('overripe')}, Unripe={reclassified.count('unripe')}, Spoiled={reclassified.count('spoiled')}")
