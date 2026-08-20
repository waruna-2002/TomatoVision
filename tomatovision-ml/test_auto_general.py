import cv2
import numpy as np

def detect_tomatoes_agro_master(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # 1. TOMATO COLOR FILTER (High-saturation Red, Orange, Yellow, Breakers)
    # Excludes: Sand, Concrete, Potatoes, Wood (S < 65)
    # =========================================================================
    mask_red = ((h_c <= 15) | (h_c >= 160)) & (s_c > 75) & (v_c > 45)
    mask_orange = (h_c > 15) & (h_c <= 28) & (s_c > 80) & (v_c > 50)
    mask_yellow = (h_c > 28) & (h_c <= 42) & (s_c > 80) & (v_c > 50)
    mask_breaker = (h_c > 42) & (h_c <= 65) & (s_c > 70) & (v_c > 50)

    # Core tomato seeds (Red + Orange + Yellow)
    tomato_core = (mask_red | mask_orange | mask_yellow).astype(np.uint8) * 255

    # Find where the true tomato fruits cluster
    kernel_group = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    dilated_core = cv2.morphologyEx(tomato_core, cv2.MORPH_CLOSE, kernel_group)
    dilated_core = cv2.dilate(dilated_core, kernel_group, iterations=1)

    clusters, _ = cv2.findContours(dilated_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_clusters = []
    for c in clusters:
        area = cv2.contourArea(c)
        if area > max(3000, 0.015 * img_area):
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / bh if bh > 0 else 0
            if ar < 3.2 and not (bw > 0.80 * w and by <= 5 and bh < 0.12 * h):
                valid_clusters.append((c, bx, by, bw, bh, area))

    # =========================================================================
    # SCENARIO A: TOMATO HEAP / CRATE EXISTS
    # =========================================================================
    if valid_clusters:
        # Build Tomato Cluster Mask
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        for _, bx, by, bw, bh, _ in valid_clusters:
            # Expand cluster bounding box slightly to capture adjacent breaker/green tomatoes
            pad_x = int(0.05 * bw)
            pad_y = int(0.05 * bh)
            x1, y1 = max(0, bx - pad_x), max(0, by - pad_y)
            x2, y2 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)
            cluster_mask[y1:y2, x1:x2] = 255

        # All tomato colors restricted strictly inside the Tomato Cluster Mask!
        all_tomatoes = (mask_red | mask_orange | mask_yellow | mask_breaker).astype(np.uint8) * 255
        all_tomatoes = cv2.bitwise_and(all_tomatoes, cluster_mask)

        # Smooth per individual tomato fruit
        clean_tomatoes = cv2.morphologyEx(all_tomatoes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        clean_tomatoes = cv2.morphologyEx(clean_tomatoes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        dist = cv2.distanceTransform(clean_tomatoes, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        # Peak detection for each fruit center
        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        dilated_dist = cv2.dilate(dist, kernel_peak)
        local_max = (dist == dilated_dist) & (dist > 5.0) & (dist > 0.08 * dist.max())

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
        raw_centers = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            d_val = dist[int(cy), int(cx)]
            raw_centers.append((cx, cy, d_val))

        raw_centers.sort(key=lambda item: item[2], reverse=True)

        suppressed = []
        min_dist_sq = 17.0 ** 2
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
            r = max(14, min(42, r))

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

            # Quality Ripeness Stage
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
    # SCENARIO B: STANDALONE SINGLE / DOUBLE TOMATO (e.g. 1 green tomato on paper)
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

            # SHAPE FILTER: Real single tomato fruit is ROUND / OVAL with high circularity and solidity!
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

        if len(detections) > 2:
            detections.sort(key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]), reverse=True)
            detections = detections[:2]

        return detections

raw_images = {
    "1. Exact Market Stall Photo (Lettuce, Cabbage, Potatoes, Sand)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg"),
    "2. Market Stall 2 (Chilies & Radishes)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg"),
    "3. Crate of Tomatoes": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"),
    "4. Single Green Tomato on Paper": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :],
}

for name, img in raw_images.items():
    res = detect_tomatoes_agro_master(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
