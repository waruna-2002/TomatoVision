import cv2
import numpy as np

img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg") # Chilies & Radishes
img2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg") # Lettuce & Potatoes

def isolate_and_detect_tomatoes(img, name):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # STEP 1: Strict Red & Orange Carotenoid Pigment Anchor (Tomato Heap Core)
    # S > 130 ensures 0% overlap with chilies (pure green), radishes (white), potatoes (brown/grey), sand (dull)
    red_orange_core = (
        ((h_c <= 15) | (h_c >= 165)) & (s_c > 130) & (v_c > 60) |
        ((h_c > 15) & (h_c <= 26)) & (s_c > 135) & (v_c > 65)
    ).astype(np.uint8) * 255

    # Filter out tiny noise and find the main tomato cluster
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
    closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"{name} -> No tomato heap found!")
        return []

    # Find the largest dense cluster of red/orange
    valid_c = [c for c in contours if cv2.contourArea(c) > 3000]
    if not valid_c:
        valid_c = contours

    main_c = max(valid_c, key=cv2.contourArea)
    
    # 2. CREATE A STRICT 2D POLYGON MASK OF THE TOMATO HEAP
    # This precisely wraps the tomato pile and cuts off green chilies, radishes, and dirt!
    heap_poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(heap_poly_mask, [cv2.convexHull(main_c)], -1, 255, -1)

    # 3. EXTRACT TOMATO FRUITS STRICTLY INSIDE THE HEAP POLYGON
    # Allow all tomato colors (red, orange, yellow, unripe green) INSIDE the heap mask
    all_tomatoes = (
        ((h_c <= 15) | (h_c >= 162)) & (s_c > 60) & (v_c > 45) |
        ((h_c > 15) & (h_c <= 28)) & (s_c > 70) & (v_c > 50) |
        ((h_c > 28) & (h_c <= 42)) & (s_c > 70) & (v_c > 50) |
        ((h_c > 42) & (h_c <= 65)) & (s_c > 55) & (v_c > 45)
    ).astype(np.uint8) * 255

    heap_tomatoes = cv2.bitwise_and(all_tomatoes, heap_poly_mask)

    clean_heap = cv2.morphologyEx(heap_tomatoes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    clean_heap = cv2.morphologyEx(clean_heap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    dist = cv2.distanceTransform(clean_heap, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    dilated_dist = cv2.dilate(dist, kernel_peak)
    local_max = (dist == dilated_dist) & (dist > 4.5) & (dist > 0.08 * dist.max())

    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
    raw_centers = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
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
    counts = {}
    annotated = img.copy()
    cv2.drawContours(annotated, [cv2.convexHull(main_c)], -1, (255, 0, 0), 3) # Blue boundary around Tomato Pile!

    for cx, cy, d_val in suppressed:
        r = int(d_val * 1.55)
        r = max(14, min(38, r))

        gx1 = max(0, int(cx - r))
        gy1 = max(0, int(cy - r))
        gx2 = min(w, int(cx + r))
        gy2 = min(h, int(cy + r))

        crop = img[gy1:gy2, gx1:gx2]
        if crop.size == 0:
            continue

        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(crop_hsv[:, :, 0])
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

        if dark_spots > 0.40:
            stage = "spoiled"
            color = (0, 0, 255)
        elif (mean_h <= 18 or mean_h >= 160):
            stage = "ripe"
            color = (0, 255, 0)
        elif 19 <= mean_h <= 28:
            stage = "overripe"
            color = (0, 165, 255)
        elif 29 <= mean_h <= 65:
            stage = "unripe"
            color = (255, 255, 0)
        else:
            stage = "ripe"
            color = (0, 255, 0)

        counts[stage] = counts.get(stage, 0) + 1
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 2)
        cv2.putText(annotated, stage, (gx1, gy1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(f"d:\\project\\TomatoVision\\tomatovision-ml\\test_annotated_{name}.jpg", annotated)
    print(f"{name} -> Total: {len(suppressed)}, Counts: {counts}")

isolate_and_detect_tomatoes(img1, "chilies_radishes")
isolate_and_detect_tomatoes(img2, "lettuce_potatoes")
