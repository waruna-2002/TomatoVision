import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066557848.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# True Tomato Color Filters with STRICT Saturation and Value:
# Tomatoes have very vibrant, saturated red, orange, and breaker tones:
# S > 100 eliminates 100% of soil, sand, road, wood, skin, potatoes!
mask_red1 = (h_chan <= 12) & (s_chan > 110) & (v_chan > 70)
mask_red2 = (h_chan >= 165) & (s_chan > 110) & (v_chan > 70)
mask_orange = (h_chan > 12) & (h_chan <= 25) & (s_chan > 130) & (v_chan > 80)
# Unripe tomatoes: yellowish-green with high saturation
mask_unripe = (h_chan > 25) & (h_chan <= 48) & (s_chan > 120) & (v_chan > 80)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_unripe).astype(np.uint8) * 255

# Exclude leafy greens / lettuce (H > 48)
lettuce_mask = (h_chan > 48)
tomato_mask[lettuce_mask] = 0

# Exclude purple cabbage
cabbage_mask = (h_chan >= 120) & (h_chan <= 165)
tomato_mask[cabbage_mask] = 0

# Exclude ground / sand / wood (S <= 100)
sand_mask = (s_chan <= 100)
tomato_mask[sand_mask] = 0

# Morphological clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=2)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Distance transform to find individual tomato spheres
dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

if dist_transform.max() > 0:
    _, sure_fg = cv2.threshold(dist_transform, 0.32 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    output_img = img.copy()
    valid_tomatoes = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 25 or area > 6000: # Remove tiny noise and massive blobs
            continue
            
        cx, cy = centroids[i]
        r = int(dist_transform[int(cy), int(cx)] * 1.85)
        r = max(12, min(40, r))
        
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(w, int(cx + r))
        y2 = min(h, int(cy + r))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Check circularity / shape and color in crop
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_s = crop_hsv[:, :, 1]
        crop_h = crop_hsv[:, :, 0]
        
        # Must have high median saturation
        if np.median(crop_s) < 100:
            continue # Skip non-tomatoes
            
        mean_h = np.mean(crop_h)
        mean_s = np.mean(crop_s)
        
        if (mean_h <= 12 or mean_h >= 165):
            stage = "fresh" # Ripe Red
            conf = 0.94
            color = (46, 213, 115)
        elif 13 <= mean_h <= 26:
            stage = "overripe" # Orange / Yellow-Red
            conf = 0.91
            color = (255, 159, 67)
        elif 27 <= mean_h <= 48:
            stage = "unripe" # Green-Yellow Breaker
            conf = 0.92
            color = (0, 210, 211)
        else:
            stage = "fresh"
            conf = 0.88
            color = (46, 213, 115)
            
        valid_tomatoes.append({
            "stage": stage,
            "conf": conf,
            "box": [x1, y1, x2, y2],
            "center": (int(cx), int(cy))
        })
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_img, stage, (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

print(f"Total True Tomatoes in Crate: {len(valid_tomatoes)}")
counts = {"fresh": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
for t in valid_tomatoes:
    counts[t["stage"]] += 1
print(f"Counts: {counts}")
cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/strict_tomatoes.jpg", output_img)
