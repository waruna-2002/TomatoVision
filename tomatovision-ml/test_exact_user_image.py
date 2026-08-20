import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
h, w = img.shape[:2]
img_area = float(w * h)

print(f"Loaded exact user image: width={w}, height={h}, area={img_area}")

from api_server import detect_tomatoes_two_stage
detections = detect_tomatoes_two_stage(img)
print(f"Current api_server.py returned {len(detections)} detections on this exact image!")
