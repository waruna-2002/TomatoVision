import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787065223962.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# Tomato Color Mask (Red, Orange, Golden-Yellow, Breaker-Green)
# Red 1
mask_red1 = (h_chan <= 15) & (s_chan > 60) & (v_chan > 60)
# Red 2
mask_red2 = (h_chan >= 165) & (s_chan > 60) & (v_chan > 60)
# Orange / Golden
mask_orange = (h_chan > 15) & (h_chan <= 38) & (s_chan > 70) & (v_chan > 70)
# Unripe Green / Breaker
mask_green = (h_chan > 38) & (h_chan <= 75) & (s_chan > 65) & (v_chan > 60)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_green).astype(np.uint8) * 255

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, kernel)
tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel)

# Find Connected Components / Contours for Tomatoes
contours, _ = cv2.findContours(tomato_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detected_tomatoes = []
output_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 120 or area > 15000: # Filter noise and huge background areas
        continue
        
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(bw) / bh
    if 0.45 <= aspect_ratio <= 2.2: # Tomato roundness ratio
        # Extract crop
        crop = img[y:y+bh, x:x+bw]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        mean_h = np.mean(hsv_crop[:, :, 0])
        mean_s = np.mean(hsv_crop[:, :, 1])
        mean_v = np.mean(hsv_crop[:, :, 2])
        
        # Check if spoiled (dark black/brown rotting spot on tomato)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray_crop < 45) / float(gray_crop.size)
        
        if dark_ratio > 0.35:
            stage = "spoiled"
            conf = 0.88
            color = (0, 0, 255)
        elif (mean_h <= 14 or mean_h >= 165) and mean_s > 60:
            stage = "fresh" # Ripe Red
            conf = 0.94
            color = (0, 255, 0)
        elif 15 <= mean_h <= 38:
            stage = "overripe" # Orange / Yellow
            conf = 0.91
            color = (0, 200, 255)
        elif 39 <= mean_h <= 75:
            stage = "unripe" # Green
            conf = 0.93
            color = (255, 200, 0)
        else:
            stage = "fresh"
            conf = 0.85
            color = (0, 255, 0)
            
        detected_tomatoes.append({
            "stage": stage,
            "conf": conf,
            "box": [x, y, x + bw, y + bh]
        })
        cv2.rectangle(output_img, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(output_img, f"{stage} {conf:.2f}", (x, max(12, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

print(f"Total Tomatoes Detected in Crate: {len(detected_tomatoes)}")
counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for t in detected_tomatoes:
    counts[t["stage"]] += 1
print(f"Ripeness Distribution: {counts}")

# Save visual verification
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/smart_detected.jpg", output_img)
print("Saved visualization to smart_detected.jpg")
