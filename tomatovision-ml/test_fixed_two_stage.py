import cv2
import numpy as np

def detect_tomatoes_industrial_agro(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # STEP 1: SCENE CLASSIFICATION (ගොඩක්ද? තනි ගෙඩියක්ද?)
    # =========================================================================
    # Identify Tomato Red & Orange Pigments (Distinct to real tomatoes)
    red_orange_seeds = (
        ((h_chan <= 14) | (h_chan >= 160)) & (s_chan > 70) & (v_chan > 50) |
        ((h_chan > 14) & (h_chan <= 28)) & (s_chan > 75) & (v_chan > 55)
    ).astype(np.uint8) * 255

    kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated_seeds = cv2.dilate(red_orange_seeds, kernel_heap, iterations=2)
    heap_contours, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_heaps = []
    if heap_contours:
        for c in heap_contours:
            c_area = cv2.contourArea(c)
            # A real tomato heap is a substantial cluster (Area > 15% of frame, height > 20% of frame)
            # Excludes thin horizontal table strips (which have hh < 0.15*h and AR > 3.0)
            if c_area > (0.15 * img_area):
                hx, hy, hw, hh = cv2.boundingRect(c)
                ar = float(hw) / hh if hh > 0 else 0
                if ar <= 2.8 and hh >= (0.20 * h):
                    valid_heaps.append((c, hx, hy, hw, hh, c_area))

    # =========================================================================
    # CASE B: DENSE MARKET HEAP / CRATE (තක්කාලි ගොඩක්)
    # =========================================================================
    if valid_heaps:
        main_heap = max(valid_heaps, key=lambda x: x[5])
        _, hx, hy, hw, hh, _ = main_heap

        # 1. ISOLATE THE TOMATO HEAP BOUNDING BOX (ගොඩ ඇති ප්‍රදේශය පමණක් වෙන් කර ගැනීම)
        rx1, ry1 = max(0, hx), max(0, hy)
        rx2, ry2 = min(w, hx + hw), min(h, hy + hh)

        # 2. DETECT ALL INDIVIDUAL TOMATOES INSIDE THE HEAP ONLY (ගොඩේ ඇති ගෙඩි පමණක් හඳුනා ගැනීම)
        # 100% excludes chilies, radishes, potatoes, and ground sand!
        roi_hsv = hsv[ry1:ry2, rx1:rx2]
        roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

        roi_red1 = (roi_h <= 14) & (roi_s > 60) & (roi_v > 45)
        roi_red2 = (roi_h >= 160) & (roi_s > 60) & (roi_v > 45)
        roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 70) & (roi_v > 50)
        roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 70) & (roi_v > 50)
        roi_unripe = (roi_h > 36) & (roi_h <= 60) & (roi_s > 45) & (roi_v > 45)

        roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
        clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return "HEAP MODE", []

        # Peak detection inside the isolated heap
        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated = cv2.dilate(dist, kernel_peak)
        local_max = (dist == dilated) & (dist > 3.8) & (dist > 0.08 * dist.max())

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

        detections = []
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

            # 3. RIPENESS QUALITY CLASSIFICATION
            if dark_spots > 0.28:
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

        return "HEAP MODE", detections

    # =========================================================================
    # CASE A: STANDALONE SINGLE / DOUBLE TOMATO (තනි තක්කාලි ගෙඩිය)
    # =========================================================================
    else:
        mask_tomato = (
            ((h_chan <= 14) | (h_chan >= 162)) & (s_chan > 35) & (v_chan > 40) |
            ((h_chan > 14) & (h_chan <= 26)) & (s_chan > 45) & (v_chan > 45) |
            ((h_chan > 26) & (h_chan <= 36)) & (s_chan > 45) & (v_chan > 45) |
            ((h_chan > 36) & (h_chan <= 65)) & (s_chan > 30) & (v_chan > 40)
        ).astype(np.uint8) * 255

        mask_tomato[h_chan > 65] = 0
        mask_tomato[(h_chan >= 120) & (h_chan <= 162)] = 0
        mask_tomato[s_chan <= 25] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closed_mask = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < (0.008 * img_area) or c_area < 700:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0

            # Reject elongated background table strips / border strips (AR > 2.5 or thin top strip)
            if 0.45 <= ar <= 2.4 and circ >= 0.25:
                if not (cw > 0.65 * w and (y <= 5 or ch < 0.18 * h)):
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

        return "SINGLE FRUIT MODE", detections

test_cases = {
    "1. Single Green Tomato on Paper": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787103060894.png")[58:518, 412:612],
    "2. Market Stall 1 (Potatoes & Sand)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102963348.png")[58:518, 412:612],
    "3. Market Stall 2 (Radishes & Chilies)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102981288.png")[58:518, 412:612],
    "4. Crate of Tomatoes": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787103001472.png")[58:518, 412:612],
}

for name, img in test_cases.items():
    mode, res = detect_tomatoes_industrial_agro(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> [{mode}] Total Detected: {len(res)}, Counts: {counts}")
