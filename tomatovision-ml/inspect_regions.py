import cv2
import numpy as np

images = {
    "Single Green Tomato": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png")[86:528, 412:612]),
    "Crate": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102274874.png")[86:528, 412:612]),
    "Market 1 (Potatoes & Sand)": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102229214.png")[86:528, 412:612]),
    "Market 2 (Radishes & Chilies)": (cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102250844.png")[86:528, 412:612]),
}

for name, img in images.items():
    print(f"\n======================================")
    print(f"Scenario: {name}, Shape: {img.shape}")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red seeds (Tomatoes)
    red_seeds = (
        ((hsv[:, :, 0] <= 13) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 60) |
        ((hsv[:, :, 0] > 13) & (hsv[:, :, 0] <= 24)) & (hsv[:, :, 1] > 125) & (hsv[:, :, 2] > 65)
    ).astype(np.uint8) * 255
    red_count = cv2.countNonZero(red_seeds)
    print(f"  Red/Orange Tomato Seed Pixels: {red_count}")
