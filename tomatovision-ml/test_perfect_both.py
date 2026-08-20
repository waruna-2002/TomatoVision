import cv2
import numpy as np

img_heap1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg") # Chilies & Radishes
img_heap2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg") # Lettuce & Potatoes
img_crate = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png") # Crate
img_single_green = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787106188869.png")[88:544, 403:597] # Single green tomato

def detect_perfect_tomatoes(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # 1. HEAP VERIFICATION (Do not mistake wooden table edge for a tomato heap!)
    # =========================================================================
    red_orange_core = (
        ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
        ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
    closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_heaps = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / bh if bh > 0 else 0
            
            # A real tomato heap must NOT be a thin top wooden table edge (by <= 0.12*h and bh < 0.18*h)
            is_table_edge = (by <= int(0.12 * h) and bh < int(0.20 * h)) or (ar > 3.0) or (bw > 0.80 * w and by <= 5)
            if area > 4500 and not is_table_edge:
                valid_heaps.append(c)

    # =========================================================================
    # SCENARIO A: TOMATO HEAP DETECTED (NO CHANGES TO HEAP LOGIC AS USER REQUESTED)
    # =========================================================================
    if valid_heaps:
        heap_mask = np.zeros((h, w), dtype=np.uint8)
        for c in valid_heaps:
            cv2.drawContours(heap_mask, [cv2.convexHull(c)], -1, 255, -1)

        all_tomatoes = (
            ((h_c <= 16) | (h_c >= 162)) & (s_c > 60) & (v_c > 45) |
            ((h_c > 16) & (h_c <= 28)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 42) & (h_c <= 65)) & (s_c > 50) & (v_c > 45)
        ).astype(np.uint8) * 255

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
    # SCENARIO B: STANDALONE SINGLE / DOUBLE TOMATO FRUIT (තනි තක්කාලි ගෙඩිය)
    # =========================================================================
    else:
        # Full Tomato Color spectrum (Red, Orange, Yellow, Green Breakers)
        mask_single = (
            ((h_c <= 16) | (h_c >= 160)) & (s_c > 35) & (v_c > 35) |
            ((h_c > 16) & (h_c <= 28)) & (s_c > 40) & (v_c > 40) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 40) & (v_c > 40) |
            ((h_c > 42) & (h_c <= 68)) & (s_c > 25) & (v_c > 35)
        ).astype(np.uint8) * 255

        # Ignore non-tomato colors (blue, purple, extreme dark, paper lines)
        mask_single[h_c > 68] = 0
        mask_single[(h_c >= 120) & (h_c <= 160)] = 0
        mask_single[s_c <= 20] = 0

        # Close inner holes in the single fruit
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        closed_mask = cv2.morphologyEx(mask_single, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        single_candidates = []

        for c in contours:
            c_area = cv2.contourArea(c)
            # Must be a solid tomato fruit, at least 1500 px or 1.5% of the frame
            if c_area < max(1200, 0.015 * img_area):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
            hull = cv2.convexHull(c)
            solidity = float(c_area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

            # SHAPE FILTER: A real single tomato fruit is ROUND / OVAL with high circularity and solidity
            if 0.50 <= ar <= 2.0 and circ >= 0.35 and solidity >= 0.75:
                # Reject top wooden table edge
                if not (cw > 0.65 * w and y <= 10 and ch < 0.20 * h):
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
                    elif 29 <= mean_h <= 68:
                        stage = "unripe"
                        conf = 0.94
                    else:
                        stage = "ripe"
                        conf = 0.88

                    single_candidates.append({
                        "class_name": stage,
                        "confidence": conf,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)],
                        "area": c_area
                    })

        if single_candidates:
            # Pick the largest single tomato fruit contour (1 solid bounding box covering the entire fruit!)
            single_candidates.sort(key=lambda item: item["area"], reverse=True)
            best = single_candidates[0]
            return [{
                "class_name": best["class_name"],
                "confidence": best["confidence"],
                "box": best["box"]
            }]

        return []

tests = [
    ("1. Market Stall (Chilies & Radishes)", img_heap1),
    ("2. Market Stall (Lettuce & Potatoes)", img_heap2),
    ("3. Crate of Tomatoes", img_crate),
    ("4. Single Green Tomato on Paper", img_single_green)
]

for name, img in tests:
    res = detect_perfect_tomatoes(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
    if len(res) <= 2:
        for r in res:
            print("  Detection:", r)
