import cv2
import numpy as np

def detect_tomatoes_ultimate_agro(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. SHAPE & CIRCULARITY DETECTION:
    # A real tomato is a 3D round sphere with distinct edges.
    # Find all spherical candidates across the image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Adaptive circle radius based on image resolution
    min_r = max(12, int(0.015 * min(h, w)))
    max_r = max(35, int(0.065 * min(h, w)))
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_r * 1.5),
        param1=50,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r
    )

    candidate_fruits = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (cx, cy, r) in circles:
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_h = float(np.mean(crop_hsv[:, :, 0]))
            mean_s = float(np.mean(crop_hsv[:, :, 1]))
            mean_v = float(np.mean(crop_hsv[:, :, 2]))

            # MUST HAVE REAL TOMATO CAROTENOID COLOR (Red, Orange, Yellow, Breaker)
            # Rejects: Potatoes (S < 60), Ground (S < 60), Dark Shadows (V < 40)
            is_red = (mean_h <= 18 or mean_h >= 160) and mean_s > 68 and mean_v > 45
            is_orange = (19 <= mean_h <= 28) and mean_s > 72 and mean_v > 50
            is_yellow = (29 <= mean_h <= 42) and mean_s > 72 and mean_v > 50
            is_unripe = (43 <= mean_h <= 65) and mean_s > 60 and mean_v > 45

            if is_red or is_orange or is_yellow or is_unripe:
                # Ripeness stage
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

                if dark_spots > 0.40:
                    stage = "spoiled"
                    conf = 0.92
                elif is_red:
                    stage = "ripe"
                    conf = 0.95
                elif is_orange:
                    stage = "overripe"
                    conf = 0.92
                elif is_yellow or is_unripe:
                    stage = "unripe"
                    conf = 0.93
                else:
                    stage = "ripe"
                    conf = 0.88

                candidate_fruits.append({
                    "class_name": stage,
                    "confidence": conf,
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "cx": cx,
                    "cy": cy
                })

    # =========================================================================
    # 2. ISOLATE THE PRIMARY TOMATO CLUSTER AREA (ගොඩ ඇති Area එක පමණක් තබා ගැනීම)
    # =========================================================================
    if len(candidate_fruits) >= 4:
        # Multiple tomatoes (Heap / Crate mode)
        # Compute bounding box of the fruit cluster
        all_cx = [f["cx"] for f in candidate_fruits]
        all_cy = [f["cy"] for f in candidate_fruits]

        # Calculate density core (25th to 75th percentile + margin)
        q25_x, q75_x = np.percentile(all_cx, 15), np.percentile(all_cx, 85)
        q25_y, q75_y = np.percentile(all_cy, 15), np.percentile(all_cy, 85)

        cluster_w = q75_x - q25_x
        cluster_h = q75_y - q25_y

        min_x = max(0, q25_x - 0.35 * cluster_w)
        max_x = min(w, q75_x + 0.35 * cluster_w)
        min_y = max(0, q25_y - 0.35 * cluster_h)
        max_y = min(h, q75_y + 0.35 * cluster_h)

        filtered = []
        for f in candidate_fruits:
            if min_x <= f["cx"] <= max_x and min_y <= f["cy"] <= max_y:
                filtered.append({
                    "class_name": f["class_name"],
                    "confidence": f["confidence"],
                    "box": f["box"]
                })
        return filtered

    elif len(candidate_fruits) >= 1:
        # Single / Double tomato mode
        return [{
            "class_name": f["class_name"],
            "confidence": f["confidence"],
            "box": f["box"]
        } for f in candidate_fruits[:2]]

    else:
        # Fallback for standalone single tomato with smooth contour
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
        fallback_res = []

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

                    fallback_res.append({
                        "class_name": stage,
                        "confidence": conf,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)]
                    })

        return fallback_res[:1]

raw_images = {
    "1. Exact Market Stall (Lettuce, Cabbage, Potatoes, Sand)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg"),
    "2. Market Stall 2 (Chilies & Radishes)": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg"),
    "3. Crate of Tomatoes": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png"),
    "4. Single Green Tomato on Paper": cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :],
}

for name, img in raw_images.items():
    res = detect_tomatoes_ultimate_agro(img)
    counts = {}
    for r in res:
        c = r["class_name"]
        counts[c] = counts.get(c, 0) + 1
    print(f"\n{name} -> Total: {len(res)}, Counts: {counts}")
