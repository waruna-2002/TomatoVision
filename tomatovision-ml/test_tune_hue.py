import cv2
import numpy as np

def classify_crop(crop):
    if crop.size == 0:
        return "ripe", 0.85
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_c = crop_hsv[:, :, 0]
    s_c = crop_hsv[:, :, 1]
    v_c = crop_hsv[:, :, 2]

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_spots = np.sum(gray_crop < 35) / float(gray_crop.size) if gray_crop.size > 0 else 0
    if dark_spots > 0.30:
        return "spoiled", 0.92

    # Dominant colored pixels (ignore highlights/shadows)
    valid_px = (s_c > 60) & (v_c > 45)
    if np.sum(valid_px) > 10:
        mean_h = np.median(h_c[valid_px])
    else:
        mean_h = np.mean(h_c)

    if (mean_h <= 11 or mean_h >= 165):
        return "ripe", 0.95
    elif 12 <= mean_h <= 24:
        return "overripe", 0.93
    elif 25 <= mean_h <= 60:
        return "unripe", 0.94
    else:
        return "ripe", 0.88

img_m = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
print("Hue distribution on market stall photo...")
