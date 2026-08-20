import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]

# Convert to HSV and Lab
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply subtle bilateral/Gaussian blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circular objects of tomato size (radius 14 to 45 px)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=24,
    param1=50,
    param2=22,
    minRadius=15,
    maxRadius=42
)

print("Hough circles found:", len(circles[0]) if circles is not None else 0)

annotated = img.copy()
valid_tomatoes = []

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Check coordinates
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r), min(h, y + r)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        mean_s = np.mean(crop_hsv[:, :, 1])
        mean_v = np.mean(crop_hsv[:, :, 2])
        
        # TOMATO COLOR VERIFICATION:
        # Tomato must have Tomato Hue: Red (H<=18 or H>=160), Orange (18<H<=28), Yellow (28<H<=42), Green Breaker (42<H<=65)
        # And must have vibrant fruit saturation S > 70
        is_tomato_color = (
            ((mean_h <= 18) | (mean_h >= 160)) and mean_s > 70 |
            (18 < mean_h <= 28) and mean_s > 75 |
            (28 < mean_h <= 42) and mean_s > 75 |
            (42 < mean_h <= 65) and mean_s > 65
        )
        
        # Sand at the bottom (y > 700) or Potatoes (x > 560 with S < 65) or Lettuce (pure green leafy)
        if y > 690:
            continue
        if x > 560 and mean_s < 70:
            continue
        if x < 185:
            continue
            
        if is_tomato_color:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

            if dark_spots > 0.40:
                stage = "spoiled"
                color = (0, 0, 255)
            elif (mean_h <= 18 or mean_h >= 160):
                stage = "ripe"
                color = (0, 255, 0)
            elif 19 <= mean_h <= 28:
                stage = "overripe"
                color = (0, 165, 255)
            elif 29 <= mean_h <= 65:
                stage = "unripe"
                color = (255, 255, 0)
            else:
                stage = "ripe"
                color = (0, 255, 0)

            valid_tomatoes.append({
                "class_name": stage,
                "confidence": 0.95,
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, stage, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_hough.jpg", annotated)
print(f"Total confirmed tomato fruits: {len(valid_tomatoes)}")
counts = {}
for t in valid_tomatoes:
    c = t["class_name"]
    counts[c] = counts.get(c, 0) + 1
print(f"Counts: {counts}")
