import cv2
import numpy as np

def detect_tomatoes_point_density(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Pure Tomato Pigment (Ripe Red, Overripe Orange, Yellow Turning)
    # S > 115 eliminates sand floor, concrete, and potatoes
    tomato_core = (
        ((h_c <= 16) | (h_c >= 165)) & (s_c > 115) & (v_c > 55) |
        ((h_c > 16) & (h_c <= 28)) & (s_c > 120) & (v_c > 60) |
        ((h_c > 28) & (h_c <= 40)) & (s_c > 120) & (v_c > 60)
    ).astype(np.uint8) * 255

    clean_seeds = cv2.morphologyEx(tomato_core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_seeds)

    fruit_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        # Ignore top 8% frame edge background border noise
        if area > 100 and cy > (0.08 * h):
            fruit_components.append((cx, cy, area))

    # =========================================================================
    # SCENARIO A: TOMATO HEAP / CRATE (Multi-fruit cluster)
    # =========================================================================
    if len(fruit_components) >= 4:
        all_cx = np.array([f[0] for f in fruit_components])
        all_cy = np.array([f[1] for f in fruit_components])

        median_x = np.median(all_cx)
        median_y = np.median(all_cy)
        std_x = np.std(all_cx)
        std_y = np.std(all_cy)

        in_cluster = (np.abs(all_cx - median_x) < 1.6 * std_x) & (np.abs(all_cy - median_y) < 1.6 * std_y)
        cluster_cx = all_cx[in_cluster]
        cluster_cy = all_cy[in_cluster]

        hx1 = max(0, int(np.min(cluster_cx) - 35))
        hy1 = max(0, int(np.min(cluster_cy) - 35))
        hx2 = min(w, int(np.max(cluster_cx) + 35))
        hy2 = min(h, int(np.max(cluster_cy) + 35))

        roi_hsv = hsv[hy1:hy2, hx1:hx2]
        roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

        roi_mask = (
            ((roi_h <= 16) | (roi_h >= 162)) & (roi_s > 65) & (roi_v > 45) |
            ((roi_h > 16) & (roi_h <= 28)) & (roi_s > 70) & (roi_v > 50) |
            ((roi_h > 28) & (roi_h <= 42)) & (roi_s > 70) & (roi_v > 50) |
            ((roi_h > 42) & (roi_h <= 65)) & (roi_s > 55) & (roi_v > 45)
        ).astype(np.uint8) * 255

        clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        dilated_dist = cv2.dilate(dist, kernel_peak)
        local_max = (dist == dilated_dist) & (dist > 5.0) & (dist > 0.08 * dist.max())

        num_labels_roi, _, _, centroids_roi = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
        raw_centers = []
        for i in range(1, num_labels_roi):
            cx, cy = centroids_roi[i]
            d_val = dist[int(cy), int(cx)]
            raw_centers.append((cx, cy, d_val))

        raw_centers.sort(key=lambda item: item[2], reverse=True)

        suppressed = []
        min_dist_sq = 18.0 ** 2
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
            r = max(14, min(38, r))

            gx1 = max(0, int(hx1 + cx - r))
            gy1 = max(0, int(hy1 + cy - r))
            gx2 = min(w, int(hx1 + cx + r))
            gy2 = min(h, int(hy1 + cy + r))

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

raw_images = {
    "1. Exact Market Stall Photo": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg"),
    "2. Market Stall 2 (Chilies & Radishes)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg"),
    "3. Crate of Tomatoes": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"),
    "4. Single Green Tomato on Paper": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :],
}

for name, img in raw_images.items():
    res = detect_tomatoes_point_density(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
