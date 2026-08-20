import cv2
import numpy as np

def detect_tomatoes_perfect(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # STEP 1: CHECK FOR STANDALONE CLOSE-UP TOMATOES (1 - 4 BIG FRUITS)
    # A single tomato is a large, continuous round/oval object (Area > 3% of frame)
    # =========================================================================
    # Mask for all tomato colors (Red, Orange, Yellow, Green/Breaker)
    # Require S > 60, V > 45 to ignore white paper and pale backgrounds
    mask_tomato_all = (
        ((h_chan <= 15) | (h_chan >= 162)) & (s_chan > 60) & (v_chan > 45) | # Red
        ((h_chan > 15) & (h_chan <= 28)) & (s_chan > 70) & (v_chan > 50) |   # Orange
        ((h_chan > 28) & (h_chan <= 38)) & (s_chan > 70) & (v_chan > 50) |   # Yellow
        ((h_chan > 38) & (h_chan <= 58)) & (s_chan > 60) & (v_chan > 45)     # Green/Unripe
    ).astype(np.uint8) * 255

    # Filter background
    mask_tomato_all[h_chan > 58] = 0
    mask_tomato_all[(h_chan >= 120) & (h_chan <= 165)] = 0
    mask_tomato_all[s_chan <= 45] = 0

    # Morphological closing to get clean complete solid fruits
    kernel_close_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    solid_fruits_mask = cv2.morphologyEx(mask_tomato_all, cv2.MORPH_CLOSE, kernel_close_large)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    solid_fruits_mask = cv2.morphologyEx(solid_fruits_mask, cv2.MORPH_OPEN, kernel_open)

    contours_standalone, _ = cv2.findContours(solid_fruits_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_standalone = []
    for c in contours_standalone:
        c_area = cv2.contourArea(c)
        if c_area > (0.025 * img_area): # Each fruit is > 2.5% of total image
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0

            # Must be round/oval fruit (aspect ratio 0.5 - 2.0, circularity >= 0.25)
            # Rejects horizontal table strips (AR > 2.2) and thin bands (cw > 0.65*w and ch < 0.20*h)
            if 0.50 <= ar <= 2.0 and circ >= 0.25:
                if not (cw > 0.60 * w and ch < 0.20 * h):
                    large_standalone.append((c, x, y, cw, ch, c_area))

    # Check Red/Orange seed density
    red_seeds = (
        ((h_chan <= 13) | (h_chan >= 165)) & (s_chan > 110) & (v_chan > 60) |
        ((h_chan > 13) & (h_chan <= 25)) & (s_chan > 125) & (v_chan > 65)
    ).astype(np.uint8) * 255
    red_seed_count = cv2.countNonZero(red_seeds)

    # IF THERE ARE 1 - 4 STANDALONE TOMATOES (AND NOT A MASSIVE MULTI-TOMATO CRATE/HEAP):
    # (A massive crate/heap has large_standalone covering a huge single blob or red_seed_count > 15000)
    if len(large_standalone) in [1, 2, 3, 4] and (len(large_standalone) == 1 or red_seed_count < 8000):
        segmented = []
        for _, x, y, cw, ch, _ in large_standalone:
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
            elif 27 <= mean_h <= 58:
                stage = "unripe"
                conf = 0.94
            else:
                stage = "ripe"
                conf = 0.88

            segmented.append({
                "class_name": stage,
                "confidence": conf,
                "box": [float(x), float(y), float(x + cw), float(y + ch)]
            })
        return "SINGLE/CLOSEUP MODE", segmented

    # =========================================================================
    # STEP 2: DENSE CRATE / MARKET TOMATO HEAP MODE
    # Find the tight Tomato Heap ROI using high-saturation Red/Orange seeds
    # STRICTLY EXCLUDES green chilies on the side, cucumbers, and ground floor!
    # =========================================================================
    kernel_seed = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_red = cv2.dilate(red_seeds, kernel_seed, iterations=2)
    c_list, _ = cv2.findContours(dilated_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not c_list:
        return "NO TOMATOES", []

    valid_c = [c for c in c_list if cv2.contourArea(c) > 2000]
    if not valid_c:
        valid_c = c_list

    main_c = max(valid_c, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(main_c)
    
    # TIGHT Bounding Box (Do not expand into surrounding chilies!)
    rx1, ry1 = max(0, rx), max(0, ry)
    rx2, ry2 = min(w, rx + rw), min(h, ry + rh)

    # Segment only inside this isolated Tomato Heap ROI
    roi_hsv = hsv[ry1:ry2, rx1:rx2]
    roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

    roi_red1 = (roi_h <= 14) & (roi_s > 80) & (roi_v > 50)
    roi_red2 = (roi_h >= 164) & (roi_s > 80) & (roi_v > 50)
    roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 90) & (roi_v > 55)
    roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 90) & (roi_v > 55)
    # Only allow unripe tomatoes INSIDE the heap with roundness and high value
    roi_unripe = (roi_h > 36) & (roi_h <= 50) & (roi_s > 80) & (roi_v > 55)

    roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
    clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return "EMPTY", []

    # Dynamic Peak Window: based on crate tomato scale (avg 18-24px)
    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated = cv2.dilate(dist, kernel_peak)
    local_max = (dist == dilated) & (dist > 4.5) & (dist > 0.08 * dist.max())

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
        r = max(10, min(36, r))

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

    return "HEAP/CRATE MODE", segmented

print("TEST 1: Single Green Tomato on Paper (media_1787075256134.png viewfinder):")
img_g = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :]
mode1, res1 = detect_tomatoes_perfect(img_g)
print(f"  {mode1} -> Total: {len(res1)}, Classes: {[r['class_name'] for r in res1]}")

print("\nTEST 2: Market Stall Photo (media_1787075584775.jpg):")
img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
mode2, res2 = detect_tomatoes_perfect(img_m)
print(f"  {mode2} -> Total: {len(res2)}, Counts: Ripe={sum(1 for r in res2 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res2 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res2 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res2 if r['class_name']=='spoiled')}")

print("\nTEST 3: Crate Photo (media_1787066931244.png):")
img_c = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
mode3, res3 = detect_tomatoes_perfect(img_c)
print(f"  {mode3} -> Total: {len(res3)}, Counts: Ripe={sum(1 for r in res3 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res3 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res3 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res3 if r['class_name']=='spoiled')}")

print("\nTEST 4: Close-Up Ripe Tomato (test_tomatoes.jpg):")
img_s = cv2.imread(r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg")
mode4, res4 = detect_tomatoes_perfect(img_s)
print(f"  {mode4} -> Total: {len(res4)}, Classes: {[r['class_name'] for r in res4]}")
