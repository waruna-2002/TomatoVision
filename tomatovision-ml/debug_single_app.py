import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787106188869.png")
# Crop viewport [y: 80 to 550, x: 400 to 600]
h_all, w_all = img.shape[:2]
viewport = img[80:550, int(w_all*0.4):int(w_all*0.6)]

import api_server
res = api_server.detect_tomatoes_two_stage(viewport)
print("Single tomato result on viewport:")
print(f"Total: {len(res)}")
for r in res:
    print(r)
