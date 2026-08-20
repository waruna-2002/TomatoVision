import cv2
import numpy as np

def detect_tomatoes_adaptive(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Broad Pigment Filtering
    core_red1 = (h_chan <= 15) & (s_chan > 45) & (v_chan > 40)
    core_red2 = (h_chan >= 162) & (s_chan > 45) & (v_chan > 40)
    core_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 55) & (v_chan > 45)
    core_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 55) & (v_chan > 45)
    core_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 45) & (v_chan > 40)

    core_mask = (core_red1 | core_red2 | core_orange | core_yellow | core_unripe).astype(np.uint8) * 255
    core_mask[h_chan > 55] = 0
    core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
    core_mask[s_chan <= 40] = 0

    # 1. Check for Close-Up Mode (Standalone individual fruits) vs Crate Mode
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_clean)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours with strict geometric tomato properties (roundness, aspect ratio, solidity)
    valid_single_fruits = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (0.015 * w * h): # ignore tiny dust specks
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        aspect_ratio = float(cw) / ch if ch > 0 else 0
        
        # Reject elongated background strips (e.g. wooden table, wall borders)
        if aspect_ratio > 2.2 or aspect_ratio < 0.45:
            continue
        if circularity < 0.30:
            continue
        if cw > 0.65 * w and ch < 0.20 * h:
            continue
            
        crop = img_bgr[y:y+ch, x:x+cw]
        if crop.size == 0:
            continue
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0

        if dark_spots > 0.30:
            stage = "spoiled"
            conf = 0.92
        elif (mean_h <= 13 or mean_h >= 164):
            stage = "ripe"
            conf = 0.95
        elif 14 <= mean_h <= 26:
            stage = "overripe"
            conf = 0.92
        elif 27 <= mean_h <= 55:
            stage = "unripe"
            conf = 0.93
        else:
            stage = "ripe"
            conf = 0.88

        valid_single_fruits.append({
            "class_name": stage,
            "confidence": conf,
            "box": [float(x), float(y), float(x + cw), float(y + ch)],
            "area": area
        })

    # If there are 1 to 5 distinct standalone tomato contours, return Close-Up Mode results
    # (Provided they account for the majority of the tomato pigment area)
    total_pigment_area = cv2.countNonZero(core_mask)
    single_fruits_area = sum(f["area"] for f in valid_single_fruits)
    
    if len(valid_single_fruits) in [1, 2, 3, 4, 5] and (single_fruits_area / max(1, total_pigment_area)) > 0.40:
        return valid_single_fruits

    # =========================================================================
    # DENSE CRATE / HEAP MODE: Local Maxima Peak Finding + Spatial NMS
    # =========================================================================
    kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    cluster_mask = cv2.dilate(core_mask, kernel_group, iterations=2)

    c_list, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c_list:
        main_cluster = max(c_list, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_cluster)
        rx1 = max(0, rx - 15)
        ry1 = max(0, ry - 15)
        rx2 = min(w, rx + rw + 15)
        ry2 = min(h, ry + rh + 15)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h

    roi_hsv = hsv[ry1:ry2, rx1:rx2]
    roi_mask = (core_mask[ry1:ry2, rx1:rx2]).copy()
    clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return []

    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    dilated = cv2.dilate(dist, kernel_peak)
    local_max = (dist == dilated) & (dist > 4.0) & (dist > 0.08 * dist.max())

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))

    raw_centers = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        d_val = dist[int(cy), int(cx)]
        raw_centers.append((cx, cy, d_val))

    raw_centers.sort(key=lambda x: x[2], reverse=True)

    suppressed = []
    min_dist_sq = 15.0 ** 2

    for c in raw_centers:
        cx, cy, d_val = c
        too_close = False
        for s in suppressed:
            if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
                too_close = True
                break
        if not too_close:
            suppressed.append((cx, cy, d_val))

    segmented = []
    for cx, cy, d_val in suppressed:
        r = int(d_val * 1.65)
        r = max(10, min(65, r))

        gx1 = max(0, int(rx1 + cx - r))
        gy1 = max(0, int(ry1 + cy - r))
        gx2 = min(w, int(rx1 + cx + r))
        gy2 = min(h, int(ry1 + cy + r))

        crop = img_bgr[gy1:gy2, gx1:gx2]
        if crop.size == 0:
            continue

        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0

        if dark_spots > 0.28:
            stage = "spoiled"
            conf = 0.92
        elif (mean_h <= 13 or mean_h >= 164):
            stage = "ripe"
            conf = 0.95
        elif 14 <= mean_h <= 26:
            stage = "overripe"
            conf = 0.92
        elif 27 <= mean_h <= 55:
            stage = "unripe"
            conf = 0.93
        else:
            stage = "ripe"
            conf = 0.88

        segmented.append({
            "class_name": stage,
            "confidence": conf,
            "box": [float(gx1), float(gy1), float(gx2), float(gy2)]
        })

    return segmented

print("1. Testing Single Green Tomato on Table:")
img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")
# Check inside viewfinder region of screenshot:
res1 = detect_tomatoes_adaptive(img1[80:550, :])
print(f"   Detected count: {len(res1)}, Classes: {[r['class_name'] for r in res1]}")

print("2. Testing Full Crate Image:")
img2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
res2 = detect_tomatoes_adaptive(img2)
print(f"   Detected count: {len(res2)}, Counts: ripe={sum(1 for r in res2 if r['class_name']=='ripe')}, unripe={sum(1 for r in res2 if r['class_name']=='unripe')}, overripe={sum(1 for r in res2 if r['class_name']=='overripe')}, spoiled={sum(1 for r in res2 if r['class_name']=='spoiled')}")
