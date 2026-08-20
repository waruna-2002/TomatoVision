import cv2
import numpy as np

def detect_tomatoes_perfect(img_bgr):
    h, w = img_bgr.shape[:2]
    min_dim = min(w, h)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # 1. Broad Tomato Pigment Mask (Anchored strictly to vibrant tomato colors, ignoring ground S<80 and green chilies H>55)
    mask_red1 = (h_chan <= 13) & (s_chan > 85) & (v_chan > 50)
    mask_red2 = (h_chan >= 165) & (s_chan > 85) & (v_chan > 50)
    mask_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 95) & (v_chan > 55)
    mask_yellow = (h_chan > 25) & (h_chan <= 36) & (s_chan > 95) & (v_chan > 55)
    mask_unripe = (h_chan > 36) & (h_chan <= 55) & (s_chan > 85) & (v_chan > 50)

    core_mask = (mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_unripe).astype(np.uint8) * 255
    core_mask[h_chan > 55] = 0
    core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
    core_mask[s_chan <= 70] = 0

    # 2. STAGE 1: ISOLATE TOMATO HEAP / CRATE ROI USING RED/ORANGE SEED ANCHOR
    # Ground floor (S < 80), green chilies (H > 25), cucumbers, beetroots have 0 overlap!
    seed_red1 = (h_chan <= 13) & (s_chan > 105) & (v_chan > 60)
    seed_red2 = (h_chan >= 165) & (s_chan > 105) & (v_chan > 60)
    seed_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 115) & (v_chan > 65)
    seed_unripe_single = (h_chan > 36) & (h_chan <= 55) & (s_chan > 95) & (v_chan > 55)

    seed_mask = (seed_red1 | seed_red2 | seed_orange | seed_unripe_single).astype(np.uint8) * 255
    
    # Check if there is any tomato in frame
    if cv2.countNonZero(seed_mask) < 200:
        return []

    # Cluster seeds into Heap/Crate ROI
    kernel_seed = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated_seeds = cv2.dilate(seed_mask, kernel_seed, iterations=2)
    c_list, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if c_list:
        valid_c = [c for c in c_list if cv2.contourArea(c) > 1500]
        main_cluster = max(valid_c if valid_c else c_list, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_cluster)
        rx1 = max(0, rx - 10)
        ry1 = max(0, ry - 10)
        rx2 = min(w, rx + rw + 10)
        ry2 = min(h, ry + rh + 10)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h

    # 3. STAGE 2: ADAPTIVE SEGMENTATION INSIDE ISOLATED TOMATO REGION
    roi_mask = (core_mask[ry1:ry2, rx1:rx2]).copy()
    clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return []

    # Adaptive Peak Window: Scales dynamically based on max distance transform
    # In Close-up (1-3 large tomatoes): dist.max() is large (e.g. 50-150px) -> window is large (35-55px)
    # In Dense Crate/Heap (70-100 tomatoes): dist.max() is small (e.g. 15-30px) -> window is small (19-23px)
    max_d = dist.max()
    ksize = int(max(19, min(55, max_d * 0.75)))
    if ksize % 2 == 0:
        ksize += 1

    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(dist, kernel_peak)
    
    # Peak threshold
    local_max = (dist == dilated) & (dist > 5.0) & (dist > 0.12 * max_d)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))

    raw_centers = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        d_val = dist[int(cy), int(cx)]
        raw_centers.append((cx, cy, d_val))

    raw_centers.sort(key=lambda x: x[2], reverse=True)

    # Spatial Non-Maximum Suppression (adaptive separation based on tomato size)
    min_sep = max(14.0, max_d * 0.65)
    min_dist_sq = min_sep ** 2

    suppressed = []
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
        r = int(d_val * 1.6)
        r = max(10, min(int(max_d * 1.8), r))

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

print("TEST 1: Market Stall Photo (media_1787075584775.jpg):")
img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
res1 = detect_tomatoes_perfect(img_m)
print(f"  Total Tomatoes in Heap: {len(res1)}, Counts: Ripe={sum(1 for r in res1 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res1 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res1 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res1 if r['class_name']=='spoiled')}")

print("\nTEST 2: Crate Photo (media_1787066931244.png):")
img_c = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
res2 = detect_tomatoes_perfect(img_c)
print(f"  Total Tomatoes in Crate: {len(res2)}, Counts: Ripe={sum(1 for r in res2 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res2 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res2 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res2 if r['class_name']=='spoiled')}")

print("\nTEST 3: Single Green Tomato on Paper (media_1787075256134.png viewfinder):")
img_g = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :]
res3 = detect_tomatoes_perfect(img_g)
print(f"  Total Green Tomatoes on Paper: {len(res3)}, Classes: {[r['class_name'] for r in res3]}")

print("\nTEST 4: Close-Up Ripe Tomato (test_tomatoes.jpg):")
img_s = cv2.imread(r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg")
res4 = detect_tomatoes_perfect(img_s)
print(f"  Total Close-Up Tomatoes: {len(res4)}, Classes: {[r['class_name'] for r in res4]}")
