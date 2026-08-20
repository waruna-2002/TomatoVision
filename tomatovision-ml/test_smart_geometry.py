import cv2
import numpy as np

def detect_tomatoes_smart_geometry(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Combine all tomato hues (Red, Orange, Yellow, Unripe Green)
    # Require decent saturation to reject white paper / grey backgrounds
    mask_red = ((h_chan <= 14) | (h_chan >= 162)) & (s_chan > 60) & (v_chan > 45)
    mask_orange = (h_chan > 14) & (h_chan <= 26) & (s_chan > 70) & (v_chan > 50)
    mask_yellow = (h_chan > 26) & (h_chan <= 36) & (s_chan > 70) & (v_chan > 50)
    mask_unripe = (h_chan > 36) & (h_chan <= 65) & (s_chan > 35) & (v_chan > 45)

    all_tomato_mask = (mask_red | mask_orange | mask_yellow | mask_unripe).astype(np.uint8) * 255

    # Filter background
    all_tomato_mask[h_chan > 65] = 0
    all_tomato_mask[(h_chan >= 120) & (h_chan <= 162)] = 0
    all_tomato_mask[s_chan <= 25] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(all_tomato_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area < (0.008 * img_area) or c_area < 800:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        ar = float(cw) / ch if ch > 0 else 0
        perimeter = cv2.arcLength(c, True)
        circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(c_area) / hull_area if hull_area > 0 else 0

        # Reject elongated background table strips / border strips (AR > 2.8, AR < 0.35, or thin strip at top edge)
        if ar > 2.8 or ar < 0.35 or (cw > 0.65 * w and (y <= 5 or ch < 0.18 * h)):
            continue

        # =====================================================================
        # CHECK: IS THIS CONTOUR A SINGLE STANDALONE TOMATO?
        # A single tomato is compact, round/oval with high circularity and high solidity!
        # (Circularity >= 0.55 and Solidity >= 0.85, or Area < 12% of total image)
        # =====================================================================
        is_single_tomato = (circ >= 0.52 and solidity >= 0.85) or (c_area < 0.09 * img_area and circ >= 0.40)

        if is_single_tomato:
            # -----------------------------------------------------------------
            # SINGLE TOMATO -> 1 SINGLE SOLID BOUNDING BOX!
            # -----------------------------------------------------------------
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

        else:
            # -----------------------------------------------------------------
            # MULTI-TOMATO CRATE / HEAP -> DISTANCE TRANSFORM PEAK EXTRACTION
            # -----------------------------------------------------------------
            heap_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(heap_mask, [c], -1, 255, -1)
            heap_roi = cv2.bitwise_and(cleaned, heap_mask)[y:y+ch, x:x+cw]

            dist = cv2.distanceTransform(heap_roi, cv2.DIST_L2, 5)
            if dist.max() == 0:
                continue

            kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(dist, kernel_peak)
            local_max = (dist == dilated) & (dist > 4.5) & (dist > 0.08 * dist.max())

            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
            raw_centers = []
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                d_val = dist[int(cy), int(cx)]
                raw_centers.append((cx, cy, d_val))

            raw_centers.sort(key=lambda item: item[2], reverse=True)

            suppressed = []
            min_dist_sq = 16.0 ** 2
            for pt in raw_centers:
                cx, cy, d_val = pt
                too_close = False
                for s in suppressed:
                    if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
                        too_close = True
                        break
                if not too_close:
                    suppressed.append((cx, cy, d_val))

            for cx, cy, d_val in suppressed:
                r = int(d_val * 1.65)
                r = max(10, min(36, r))

                gx1 = max(0, int(x + cx - r))
                gy1 = max(0, int(y + cy - r))
                gx2 = min(w, int(x + cx + r))
                gy2 = min(h, int(y + cy + r))

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

raw_images = {
    "1. Single Green Tomato": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102573801.png")[58:518, 412:612],
    "2. Market Stall Photo": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg"),
    "3. Crate Photo": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"),
}

for name, img in raw_images.items():
    res = detect_tomatoes_smart_geometry(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total Detected: {len(res)}, Counts: {counts}")
