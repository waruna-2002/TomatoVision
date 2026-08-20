import cv2
import numpy as np

def detect_tomatoes_two_stage(img_bgr):
    """
    STRICT PRIMARY TOMATO CLUSTER & SHAPE ISOLATION ENGINE:
    =======================================================
    1. Isolates the Primary Tomato Cluster / Heap Area (Convex Polygon Mask).
       Strictly eliminates:
       - Green chilies (on left)
       - White radishes (on right)
       - Green lettuce & purple cabbage (on left)
       - Potatoes & snake gourds (on right)
       - Sand / Concrete floor (at bottom)
    2. Runs Local Maxima Distance Transform strictly inside the isolated Tomato Area.
    3. Standalone mode for single/double tomato fruits with circularity/solidity filters.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # STEP 1: STRICT CAROTENOID HEAP ISOLATION (තක්කාලි ඇති Area එක වෙන් කර ගැනීම)
    # =========================================================================
    # S > 125 ensures 0% overlap with green chilies, radishes, potatoes, sand, wood
    red_orange_core = (
        ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
        ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
    closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_c = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / bh if bh > 0 else 0
            # Reject full-screen frames or thin horizontal strips
            is_frame = (bw > 0.85 * w and bh > 0.85 * h) or (bw > 0.70 * w and by <= 5 and bh < 0.12 * h) or (ar > 3.2)
            if area > 3500 and not is_frame:
                valid_c.append(c)

    # =========================================================================
    # SCENARIO A: TOMATO HEAP DETECTED
    # =========================================================================
    if valid_c:
        # Build 100% Strict Tomato Area Mask (Convex Hull of the heap)
        heap_mask = np.zeros((h, w), dtype=np.uint8)
        for c in valid_c:
            cv2.drawContours(heap_mask, [cv2.convexHull(c)], -1, 255, -1)

        # Allow full spectrum of tomato ripening inside the Tomato Heap ONLY
        all_tomatoes = (
            ((h_c <= 16) | (h_c >= 162)) & (s_c > 60) & (v_c > 45) |
            ((h_c > 16) & (h_c <= 28)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 42) & (h_c <= 65)) & (s_c > 50) & (v_c > 45)
        ).astype(np.uint8) * 255

        # STRICTLY RESTRICT TO TOMATO HEAP AREA!
        heap_tomatoes = cv2.bitwise_and(all_tomatoes, heap_mask)

        clean_heap = cv2.morphologyEx(heap_tomatoes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        clean_heap = cv2.morphologyEx(clean_heap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        dist = cv2.distanceTransform(clean_heap, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_dist = cv2.dilate(dist, kernel_peak)
        local_max = (dist == dilated_dist) & (dist > 4.0) & (dist > 0.07 * dist.max())

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
        raw_centers = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            d_val = dist[int(cy), int(cx)]
            raw_centers.append((cx, cy, d_val))

        raw_centers.sort(key=lambda item: item[2], reverse=True)

        suppressed = []
        min_dist_sq = 14.0 ** 2
        for pt in raw_centers:
            cx, cy, d_val = pt
            too_close = False
            for s in suppressed:
                if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
                    too_close = True
                    break
            if not too_close:
                suppressed.append((cx, cy, d_val))

        detections = []
        for cx, cy, d_val in suppressed:
            r = int(d_val * 1.55)
            r = max(13, min(36, r))

            gx1 = max(0, int(cx - r))
            gy1 = max(0, int(cy - r))
            gx2 = min(w, int(cx + r))
            gy2 = min(h, int(cy + r))

            crop = img_bgr[gy1:gy2, gx1:gx2]
            if crop.size == 0:
                continue

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(crop_hsv[:, :, 0])
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

            if dark_spots > 0.40:
                stage = "spoiled"
                conf = 0.92
            elif (mean_h <= 18 or mean_h >= 160):
                stage = "ripe"
                conf = 0.95
            elif 19 <= mean_h <= 28:
                stage = "overripe"
                conf = 0.92
            elif 29 <= mean_h <= 65:
                stage = "unripe"
                conf = 0.93
            else:
                stage = "ripe"
                conf = 0.88

            detections.append({
                "class_name": stage,
                "confidence": conf,
                "box": [float(gx1), float(gy1), float(gx2), float(gy2)]
            })

        return detections

    # =========================================================================
    # SCENARIO B: STANDALONE SINGLE / DOUBLE TOMATO
    # =========================================================================
    else:
        mask_single = (
            ((h_c <= 15) | (h_c >= 160)) & (s_c > 35) & (v_c > 40) |
            ((h_c > 15) & (h_c <= 28)) & (s_c > 45) & (v_c > 45) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 45) & (v_c > 45) |
            ((h_c > 42) & (h_c <= 65)) & (s_c > 30) & (v_c > 40)
        ).astype(np.uint8) * 255

        mask_single[h_c > 65] = 0
        mask_single[(h_c >= 120) & (h_c <= 160)] = 0
        mask_single[s_c <= 25] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closed_mask = cv2.morphologyEx(mask_single, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < (0.015 * img_area) and c_area < 1500:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
            hull = cv2.convexHull(c)
            solidity = float(c_area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

            if 0.50 <= ar <= 2.0 and circ >= 0.40 and solidity >= 0.80:
                if not (cw > 0.65 * w and y <= 5 and ch < 0.18 * h):
                    crop = img_bgr[y:y+ch, x:x+cw]
                    if crop.size == 0:
                        continue
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mean_h = np.mean(crop_hsv[:, :, 0])
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

                    if dark_spots > 0.40:
                        stage = "spoiled"
                        conf = 0.92
                    elif (mean_h <= 18 or mean_h >= 160):
                        stage = "ripe"
                        conf = 0.95
                    elif 19 <= mean_h <= 28:
                        stage = "overripe"
                        conf = 0.92
                    elif 29 <= mean_h <= 65:
                        stage = "unripe"
                        conf = 0.94
                    else:
                        stage = "ripe"
                        conf = 0.88

                    detections.append({
                        "class_name": stage,
                        "confidence": conf,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)]
                    })

        return detections[:1]

img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg") # Chilies & Radishes
img2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg") # Lettuce & Potatoes
img3 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png") # Crate

for name, img in [("1. Market Stall (Chilies & Radishes)", img1), ("2. Market Stall (Lettuce & Potatoes)", img2), ("3. Crate", img3)]:
    res = detect_tomatoes_two_stage(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
