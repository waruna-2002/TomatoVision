import cv2
import numpy as np

def detect_tomatoes_agro_vision(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Broad Pigment Filtering (Red, Orange, Yellow, Breaker Unripe)
    core_red1 = (h_chan <= 14) & (s_chan > 60) & (v_chan > 40)
    core_red2 = (h_chan >= 162) & (s_chan > 60) & (v_chan > 40)
    core_orange = (h_chan > 14) & (h_chan <= 28) & (s_chan > 70) & (v_chan > 45)
    core_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 70) & (v_chan > 45)
    core_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 60) & (v_chan > 40)

    core_mask = (core_red1 | core_red2 | core_orange | core_yellow | core_unripe).astype(np.uint8) * 255
    core_mask[h_chan > 55] = 0
    core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
    core_mask[s_chan <= 50] = 0

    # 1. Close-up Single / Multi-Fruit Detector (Shape & Circularity Filter)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_clean)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_single_fruits = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < (0.012 * w * h):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        aspect_ratio = float(cw) / ch if ch > 0 else 0

        # Reject elongated background strips (e.g. table edges, floor boundaries)
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
            "raw_class": stage,
            "confidence": conf,
            "box": [float(x), float(y), float(x + cw), float(y + ch)],
            "area": area
        })

    total_pigment_area = cv2.countNonZero(core_mask)
    single_fruits_area = sum(f["area"] for f in valid_single_fruits)

    if len(valid_single_fruits) in [1, 2, 3, 4, 5] and (single_fruits_area / max(1, total_pigment_area)) > 0.35:
        return valid_single_fruits

    # =========================================================================
    # 2. DENSE TOMATO HEAP / CRATE ISOLATION
    # Uses High-Saturation Red/Orange Tomato Seeds (S > 105) to strictly anchor the Tomato Heap ROI
    # Rejecting surrounding green vegetables, cucumbers, eggplants, and dull ground floor.
    # =========================================================================
    seed_red1 = (h_chan <= 13) & (s_chan > 105) & (v_chan > 60)
    seed_red2 = (h_chan >= 165) & (s_chan > 105) & (v_chan > 60)
    seed_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 115) & (v_chan > 65)
    seed_mask = (seed_red1 | seed_red2 | seed_orange).astype(np.uint8) * 255

    kernel_seed_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_seeds = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel_seed_close)
    kernel_seed_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated_seeds = cv2.dilate(closed_seeds, kernel_seed_dilate, iterations=2)

    c_list, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c_list:
        valid_c = [c for c in c_list if cv2.contourArea(c) > 2000]
        main_cluster = max(valid_c if valid_c else c_list, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_cluster)
        rx1 = max(0, rx - 10)
        ry1 = max(0, ry - 10)
        rx2 = min(w, rx + rw + 10)
        ry2 = min(h, ry + rh + 10)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h

    roi_hsv = hsv[ry1:ry2, rx1:rx2]
    roi_h = roi_hsv[:, :, 0]
    roi_s = roi_hsv[:, :, 1]
    roi_v = roi_hsv[:, :, 2]

    roi_red1 = (roi_h <= 14) & (roi_s > 75) & (roi_v > 50)
    roi_red2 = (roi_h >= 164) & (roi_s > 75) & (roi_v > 50)
    roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 85) & (roi_v > 55)
    roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 85) & (roi_v > 55)
    roi_unripe = (roi_h > 36) & (roi_h <= 50) & (roi_s > 75) & (roi_v > 50)

    roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
    clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return []

    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
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
            "raw_class": stage,
            "confidence": conf,
            "box": [float(gx1), float(gy1), float(gx2), float(gy2)]
        })

    return segmented

# TEST 1: Market Stall Pile (media_1787075584775.jpg)
img_market = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
res_market = detect_tomatoes_agro_vision(img_market)
print(f"1. Market Stall Photo -> Total: {len(res_market)}, Counts: Ripe={sum(1 for r in res_market if r['class_name']=='ripe')}, Unripe={sum(1 for r in res_market if r['class_name']=='unripe')}, Overripe={sum(1 for r in res_market if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res_market if r['class_name']=='spoiled')}")

# TEST 2: Crate Photo (media_1787066931244.png)
img_crate = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
res_crate = detect_tomatoes_agro_vision(img_crate)
print(f"2. Crate Photo -> Total: {len(res_crate)}, Counts: Ripe={sum(1 for r in res_crate if r['class_name']=='ripe')}, Unripe={sum(1 for r in res_crate if r['class_name']=='unripe')}, Overripe={sum(1 for r in res_crate if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res_crate if r['class_name']=='spoiled')}")

# TEST 3: Single Tomato Close-up (test_tomatoes.jpg)
img_single = cv2.imread(r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg")
res_single = detect_tomatoes_agro_vision(img_single)
print(f"3. Single Tomato Close-up -> Total: {len(res_single)}, Counts: {[r['class_name'] for r in res_single]}")
