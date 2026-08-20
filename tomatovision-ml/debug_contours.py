import cv2
import numpy as np

img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102963348.png")[58:518, 412:612]
img2 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102981288.png")[58:518, 412:612]
img3 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787103001472.png")[58:518, 412:612]

for idx, img in enumerate([img1, img2, img3], 1):
    h, w = img.shape[:2]
    img_area = float(w * h)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # In user uploaded screenshots, because they have HUD lines, let's look at the red seeds
    red_orange_seeds = (
        ((h_chan <= 14) | (h_chan >= 160)) & (s_chan > 70) & (v_chan > 50) |
        ((h_chan > 14) & (h_chan <= 28)) & (s_chan > 75) & (v_chan > 55)
    ).astype(np.uint8) * 255

    kernel_heap = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated_seeds = cv2.dilate(red_orange_seeds, kernel_heap, iterations=2)
    heap_contours, _ = cv2.findContours(dilated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\nPhoto {idx}: img_area={img_area}, total heap contours={len(heap_contours)}")
    for i, c in enumerate(heap_contours):
        c_area = cv2.contourArea(c)
        hx, hy, hw, hh = cv2.boundingRect(c)
        ar = float(hw) / hh if hh > 0 else 0
        print(f"  Contour {i}: Area={c_area:.0f} ({(c_area/img_area)*100:.1f}%), BBox=[{hx},{hy},{hw},{hh}], AR={ar:.2f}")
