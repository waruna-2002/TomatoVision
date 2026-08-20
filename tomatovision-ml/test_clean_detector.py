import cv2
import numpy as np

def detect_tomatoes(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # 1. Strict Tomato Anchor Seed (Vibrant Red & Orange)
    # Excludes ground floor (S < 100), green chilies (H > 25), cucumbers, beetroots
    seed_red1 = (h_chan <= 13) & (s_chan > 100) & (v_chan > 60)
    seed_red2 = (h_chan >= 165) & (s_chan > 100) & (v_chan > 60)
    seed_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 115) & (v_chan > 65)
    
    # Also include breaker / green unripe IF it is a single isolated fruit on table
    seed_green = (h_chan > 28) & (h_chan <= 55) & (s_chan > 90) & (v_chan > 60)

    # 2. Check if this is a Single/Few Fruit Close-Up
    clean_green = cv2.morphologyEx(seed_green.astype(np.uint8)*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    clean_red = cv2.morphologyEx((seed_red1 | seed_red2 | seed_orange).astype(np.uint8)*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    combined_clean = cv2.bitwise_or(clean_green, clean_red)
    contours_all, _ = cv2.findContours(combined_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for large individual tomato contours (each > 3% of image area)
    large_single_tomatoes = []
    for c in contours_all:
        c_area = cv2.contourArea(c)
        if c_area > (0.03 * img_area):
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
            
            # Must be organic round / oval shape (reject elongated table strips AR > 2.2)
            if 0.45 <= ar <= 2.2 and circ >= 0.28:
                if not (cw > 0.65 * w and ch < 0.20 * h):
                    large_single_tomatoes.append((c, x, y, cw, ch, c_area))

    # If there are 1-4 large distinct standalone tomatoes in the frame (Close-Up Mode):
    if len(large_single_tomatoes) in [1, 2, 3, 4] and cv2.countNonZero(clean_red) < (0.15 * img_area):
        segmented = []
        for _, x, y, cw, ch, _ in large_single_tomatoes:
            crop = img_bgr[y:y+ch, x:x+cw]
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
                "box": [float(x), float(y), float(x + cw), float(y + ch)]
            })
        return "CLOSEUP MODE", segmented

    # =========================================================================
    # 3. DENSE CRATE / MARKET HEAP MODE (Multi-Tomato Cluster)
    # Find the main Tomato Cluster/Crate ROI using Red/Orange seeds
    # =========================================================================
    seed_mask = (seed_red1 | seed_red2 | seed_orange).astype(np.uint8) * 255
    closed_seeds = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dilated_seeds = cv2.dilate(closed_seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)

    c_list, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c_list:
        valid_c = [c for c in c_list if cv2.contourArea(c) > 2500]
        main_cluster = max(valid_c if valid_c else c_list, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_cluster)
        rx1 = max(0, rx - 10)
        ry1 = max(0, ry - 10)
        rx2 = min(w, rx + rw + 10)
        ry2 = min(h, ry + rh + 10)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h

    # Segment only inside the isolated Tomato Heap ROI
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
        return "EMPTY", []

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

print("TEST 1: Market Stall Photo (media_1787075584775.jpg):")
img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
mode1, res1 = detect_tomatoes(img_m)
print(f"  {mode1} -> Total: {len(res1)}, Counts: Ripe={sum(1 for r in res1 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res1 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res1 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res1 if r['class_name']=='spoiled')}")

print("\nTEST 2: Crate Photo (media_1787066931244.png):")
img_c = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
mode2, res2 = detect_tomatoes(img_c)
print(f"  {mode2} -> Total: {len(res2)}, Counts: Ripe={sum(1 for r in res2 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res2 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res2 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res2 if r['class_name']=='spoiled')}")

print("\nTEST 3: Single Tomato Close-up (test_tomatoes.jpg):")
img_s = cv2.imread(r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg")
mode3, res3 = detect_tomatoes(img_s)
print(f"  {mode3} -> Total: {len(res3)}, Counts: {[r['class_name'] for r in res3]}")

print("\nTEST 4: Single Green Tomato on Paper (media_1787075256134.png viewfinder area):")
img_g = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :]
mode4, res4 = detect_tomatoes(img_g)
print(f"  {mode4} -> Total: {len(res4)}, Counts: {[r['class_name'] for r in res4]}")
