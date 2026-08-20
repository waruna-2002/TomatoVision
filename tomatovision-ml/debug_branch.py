import cv2
import numpy as np

img_bgr = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")
h, w = img_bgr.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

red_seeds = (
    ((h_chan <= 13) | (h_chan >= 165)) & (s_chan > 110) & (v_chan > 60) |
    ((h_chan > 13) & (h_chan <= 25)) & (s_chan > 130) & (v_chan > 70)
).astype(np.uint8) * 255

kernel_seed = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilated_red = cv2.dilate(red_seeds, kernel_seed, iterations=2)
c_list, _ = cv2.findContours(dilated_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_clusters = []
if c_list:
    for c in c_list:
        c_area = cv2.contourArea(c)
        if c_area < 3000:
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        ar = float(cw) / ch if ch > 0 else 0
        print(f"Cluster: Area={c_area}, BBox=[{cx},{cy},{cw},{ch}], AR={ar:.2f}, cond={not (cw > 0.60 * w and ch < 0.20 * h)}")
        if 0.45 <= ar <= 2.2 and not (cw > 0.60 * w and ch < 0.20 * h):
            valid_clusters.append((c, cx, cy, cw, ch, c_area))

print(f"valid_clusters count: {len(valid_clusters)}")
