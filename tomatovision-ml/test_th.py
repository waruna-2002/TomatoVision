import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066557848.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

# Tomato Mask
mask_red1 = (h_chan <= 14) & (s_chan > 90) & (v_chan > 60)
mask_red2 = (h_chan >= 165) & (s_chan > 90) & (v_chan > 60)
mask_orange = (h_chan > 14) & (h_chan <= 28) & (s_chan > 100) & (v_chan > 70)
mask_unripe = (h_chan > 28) & (h_chan <= 50) & (s_chan > 100) & (v_chan > 70)

tomato_mask = (mask_red1 | mask_red2 | mask_orange | mask_unripe).astype(np.uint8) * 255

# Exclude lettuce and cabbage
tomato_mask[h_chan > 50] = 0
tomato_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
# Exclude low saturation floor/sand/boxes
tomato_mask[s_chan <= 85] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_OPEN, kernel, iterations=1)
dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

for th in [0.20, 0.25, 0.30]:
    _, sure_fg = cv2.threshold(dist_transform, th * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    tomatoes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 8:
            continue
        cx, cy = centroids[i]
        tomatoes.append((int(cx), int(cy)))
    print(f"Threshold {th}: {len(tomatoes)} seeds found.")
