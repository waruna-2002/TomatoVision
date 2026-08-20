import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787106188869.png")
# The camera frame is at x: 403 to 597, y: 88 to 544
camera_frame = img[88:544, 403:597]

import api_server
res = api_server.detect_tomatoes_two_stage(camera_frame)
print("Camera frame result:")
print(f"Total: {len(res)}")
for r in res:
    print(r)
