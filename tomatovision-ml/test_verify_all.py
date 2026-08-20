import cv2
import numpy as np

def detect_tomatoes_final(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Red & Orange Seeds (Strictly anchor Tomato Heaps, 0% overlap with green chilies, radishes, potatoes, sand)
    red_seeds = (
        ((h_chan <= 13) | (h_chan >= 165)) & (s_chan > 105) & (v_chan > 60) |
        ((h_chan > 13) & (h_chan <= 25)) & (s_chan > 120) & (v_chan > 65)
    ).astype(np.uint8) * 255
    red_seed_count = cv2.countNonZero(red_seeds)

    # -------------------------------------------------------------------------
    # BRANCH 1: TOMATO HEAP / CRATE (When a Red/Orange Tomato Cluster Exists)
    # -------------------------------------------------------------------------
    if red_seed_count > (0.012 * img_area) or red_seed_count > 1000:
        kernel_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_red = cv2.dilate(red_seeds, kernel_seed, iterations=2)
        c_list, _ = cv2.findContours(dilated_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_c = [c for c in c_list if cv2.contourArea(c) > 1200]
        if not valid_c:
            valid_c = c_list

        main_c = max(valid_c, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_c)
        rx1, ry1 = max(0, rx), max(0, ry)
        rx2, ry2 = min(w, rx + rw), min(h, ry + rh)

        # Pixels belonging to tomatoes strictly inside this isolated Heap ROI
        roi_hsv = hsv[ry1:ry2, rx1:rx2]
        roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

        roi_red1 = (roi_h <= 14) & (roi_s > 80) & (roi_v > 50)
        roi_red2 = (roi_h >= 164) & (roi_s > 80) & (roi_v > 50)
        roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 85) & (roi_v > 55)
        roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 85) & (roi_v > 55)
        roi_unripe = (roi_h > 36) & (roi_h <= 55) & (roi_s > 80) & (roi_v > 55)

        roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
        clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return []

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
        min_dist_sq = 16.0 ** 2

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

            segmented.append({
                "class_name": stage,
                "confidence": conf,
                "box": [float(gx1), float(gy1), float(gx2), float(gy2)]
            })

        return segmented

    # -------------------------------------------------------------------------
    # BRANCH 2: STANDALONE CLOSE-UP SINGLE / DOUBLE TOMATO (e.g. 1 green tomato on paper)
    # -------------------------------------------------------------------------
    else:
        mask_tomato = (
            ((h_chan <= 15) | (h_chan >= 162)) & (s_chan > 35) & (v_chan > 40) |
            ((h_chan > 15) & (h_chan <= 28)) & (s_chan > 40) & (v_chan > 45) |
            ((h_chan > 28) & (h_chan <= 38)) & (s_chan > 40) & (v_chan > 45) |
            ((h_chan > 38) & (h_chan <= 65)) & (s_chan > 35) & (v_chan > 40)
        ).astype(np.uint8) * 255

        mask_tomato[h_chan > 65] = 0
        mask_tomato[(h_chan >= 120) & (h_chan <= 165)] = 0
        mask_tomato[s_chan <= 25] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closed_mask = cv2.morphologyEx(mask_tomato, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented = []

        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < (0.010 * img_area) or c_area < 800:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0

            # Reject elongated table strips / borders
            if 0.50 <= ar <= 2.2 and circ >= 0.25:
                if not (cw > 0.70 * w and ch < 0.20 * h):
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

                    segmented.append({
                        "class_name": stage,
                        "confidence": conf,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)]
                    })

        return segmented

images = {
    "1. Single Green Tomato": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png")[58:518, 412:612]),
    "2. Crate": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102274874.png")[58:518, 412:612]),
    "3. Market 1 (Potatoes & Sand)": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102229214.png")[58:518, 412:612]),
    "4. Market 2 (Radishes & Chilies)": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102250844.png")[58:518, 412:612]),
}

for name, img in images.items():
    res = detect_tomatoes_final(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
