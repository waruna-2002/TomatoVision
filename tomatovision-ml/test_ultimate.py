import cv2
import numpy as np

def detect_tomatoes_ultimate(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Red/Orange Anchor Seed (0% overlap with green chilies, cucumbers, beetroots, sand floor)
    seed_red1 = (h_chan <= 13) & (s_chan > 110) & (v_chan > 60)
    seed_red2 = (h_chan >= 165) & (s_chan > 110) & (v_chan > 60)
    seed_orange = (h_chan > 13) & (h_chan <= 25) & (s_chan > 125) & (v_chan > 65)

    red_seed_mask = (seed_red1 | seed_red2 | seed_orange).astype(np.uint8) * 255
    red_pixel_count = cv2.countNonZero(red_seed_mask)

    # SCENARIO A: A Red/Orange Tomato Cluster Exists (Crate or Market Heap or Ripe Tomatoes)
    if red_pixel_count > 500:
        # Group red/orange seeds into the Tomato Pile ROI
        kernel_seed = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed_seeds = cv2.morphologyEx(red_seed_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        dilated_seeds = cv2.dilate(closed_seeds, kernel_seed, iterations=2)

        c_list, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_c = [c for c in c_list if cv2.contourArea(c) > 2500]
        main_cluster = max(valid_c if valid_c else c_list, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(main_cluster)
        rx1, ry1 = max(0, rx - 12), max(0, ry - 12)
        rx2, ry2 = min(w, rx + rw + 12), min(h, ry + rh + 12)

        # Tomato pigments strictly inside this isolated ROI
        roi_hsv = hsv[ry1:ry2, rx1:rx2]
        roi_h, roi_s, roi_v = roi_hsv[:, :, 0], roi_hsv[:, :, 1], roi_hsv[:, :, 2]

        roi_red1 = (roi_h <= 14) & (roi_s > 80) & (roi_v > 50)
        roi_red2 = (roi_h >= 164) & (roi_s > 80) & (roi_v > 50)
        roi_orange = (roi_h > 14) & (roi_h <= 26) & (roi_s > 90) & (roi_v > 55)
        roi_yellow = (roi_h > 26) & (roi_h <= 36) & (roi_s > 90) & (roi_v > 55)
        roi_unripe = (roi_h > 36) & (roi_h <= 50) & (roi_s > 80) & (roi_v > 50)

        roi_mask = (roi_red1 | roi_red2 | roi_orange | roi_yellow | roi_unripe).astype(np.uint8) * 255
        clean_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dist = cv2.distanceTransform(clean_roi, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return []

        # Peak Finding
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

            gx1, gy1 = max(0, int(rx1 + cx - r)), max(0, int(ry1 + cy - r))
            gx2, gy2 = min(w, int(rx1 + cx + r)), min(h, int(ry1 + cy + r))

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

        return segmented

    # SCENARIO B: NO RED/ORANGE HEAP FOUND -> Check for Standalone Single Green/Unripe Tomato
    else:
        unripe_mask = (h_chan > 28) & (h_chan <= 55) & (s_chan > 60) & (v_chan > 50)
        unripe_clean = cv2.morphologyEx(unripe_mask.astype(np.uint8)*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        unripe_closed = cv2.morphologyEx(unripe_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        contours, _ = cv2.findContours(unripe_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < (0.015 * img_area):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

            # Reject non-tomato background strips
            if 0.45 <= ar <= 2.2 and circ >= 0.25:
                if not (cw > 0.65 * w and ch < 0.20 * h):
                    segmented.append({
                        "class_name": "unripe",
                        "confidence": 0.93,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)]
                    })
        return segmented

print("1. Market Stall Photo (media_1787075584775.jpg):")
img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
res1 = detect_tomatoes_ultimate(img_m)
print(f"   Total Tomatoes in Heap: {len(res1)}, Counts: Ripe={sum(1 for r in res1 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res1 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res1 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res1 if r['class_name']=='spoiled')}")

print("\n2. Crate Photo (media_1787066931244.png):")
img_c = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787066931244.png")
res2 = detect_tomatoes_ultimate(img_c)
print(f"   Total Tomatoes in Crate: {len(res2)}, Counts: Ripe={sum(1 for r in res2 if r['class_name']=='ripe')}, Unripe={sum(1 for r in res2 if r['class_name']=='unripe')}, Overripe={sum(1 for r in res2 if r['class_name']=='overripe')}, Spoiled={sum(1 for r in res2 if r['class_name']=='spoiled')}")

print("\n3. Single Green Tomato on Paper (media_1787075256134.png viewfinder):")
img_g = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png")[80:550, :]
res3 = detect_tomatoes_ultimate(img_g)
print(f"   Total Green Tomatoes: {len(res3)}, Classes: {[r['class_name'] for r in res3]}")
