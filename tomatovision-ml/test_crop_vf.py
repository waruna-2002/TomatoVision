import cv2
import os
import numpy as np

img_files = [
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png", # Single green tomato on paper
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102274874.png", # Crate of tomatoes
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102229214.png", # Market with potatoes & sand
    r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102250844.png", # Market with radishes & chilies
]

def crop_viewfinder_if_screenshot(img):
    h, w = img.shape[:2]
    if w > 1200 and h > 700:
        return img[80:550, int(w*0.38):int(w*0.62)]
    return img

for f in img_files:
    raw = cv2.imread(f)
    fname = os.path.basename(f)
    print(f"\n======================================")
    print(f"File: {fname}, Raw Shape: {raw.shape}")
    vf = crop_viewfinder_if_screenshot(raw)
    print(f"Viewfinder Crop Shape: {vf.shape}")
