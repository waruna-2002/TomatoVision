import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066557848.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# Tomato Color Mask
mask_red1 = (h_chan <= 13) & (s_chan > 90) & (v_chan > 70)
mask_red2 = (h_chan >= 165) & (s_chan > 90) & (v_chan > 70)
mask_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 115) & (v_chan > 80)
mask_unripe = (h_chan > 25) & (h_chan <= 46) & (s_chan > 115) & (v_chan > 80)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_unripe).astype(np.uint8) * 255
tomato_mask[h_chan > 46] = 0
tomato_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
tomato_mask[s_chan <= 90] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=1)
dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

tomatoes = []
output_img = img.copy()

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < 8:
        continue
    cx, cy = centroids[i]
    r = int(dist_transform[int(cy), int(cx)] * 1.8)
    r = max(10, min(35, r))
    
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
    
    crop = img[y1:y2, x1:x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if np.median(crop_hsv[:, :, 1]) < 90:
        continue
        
    mean_h = np.mean(crop_hsv[:, :, 0])
    if (mean_h <= 13 or mean_h >= 165):
        stage = "fresh"
        color = (46, 213, 115)
    elif 14 <= mean_h <= 26:
        stage = "overripe"
        color = (255, 159, 67)
    elif 27 <= mean_h <= 46:
        stage = "unripe"
        color = (0, 210, 211)
    else:
        stage = "fresh"
        color = (46, 213, 115)
        
    tomatoes.append((stage, (x1, y1, x2, y2), (int(cx), int(cy))))
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

print(f"Total tomatoes strictly filtered: {len(tomatoes)}")
print("Coordinates distribution:")
for t in tomatoes[:15]:
    print(f"  {t[0]} at {t[2]}")

cv2.imwrite("d:/project/TomatoVision/tomatovision-ml/checked_coords.jpg", output_img)
