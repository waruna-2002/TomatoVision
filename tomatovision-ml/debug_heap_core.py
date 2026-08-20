import cv2
import numpy as np

# Let us load the market photos
img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102963348.png")[58:518, 412:612]
img2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102981288.png")[58:518, 412:612]
img3 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787103001472.png")[58:518, 412:612]

def find_tomato_heap(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # In a market heap, tomatoes are clustered in a central region with rich Red/Orange pigments:
    # Tomato Red: H in [0, 14] or [160, 180], S > 85, V > 55
    # Tomato Orange: H in [15, 26], S > 90, V > 60
    # Tomato Yellow/Breaker: H in [27, 36], S > 85, V > 60
    
    red_orange = (
        ((h_c <= 14) | (h_c >= 160)) & (s_c > 85) & (v_c > 55) |
        ((h_c > 14) & (h_c <= 26)) & (s_c > 90) & (v_c > 60)
    ).astype(np.uint8) * 255

    # Density filter: Tomato heaps have high local density of red/orange
    kernel_blur = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    density = cv2.morphologyEx(red_orange, cv2.MORPH_CLOSE, kernel_blur)
    density = cv2.dilate(density, kernel_blur, iterations=1)

    contours, _ = cv2.findContours(density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    valid = [c for c in contours if cv2.contourArea(c) > 2000]
    if not valid:
        return None, 0

    main_heap = max(valid, key=cv2.contourArea)
    hx, hy, hw, hh = cv2.boundingRect(main_heap)
    return (hx, hy, hw, hh), cv2.contourArea(main_heap)

for i, img in enumerate([img1, img2, img3], 1):
    bbox, area = find_tomato_heap(img)
    print(f"Market Photo {i} -> Tomato Heap Bounding Box: {bbox}, Area: {area}")
