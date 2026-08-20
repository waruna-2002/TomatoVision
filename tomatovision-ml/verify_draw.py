import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")

from test_high_sat import detect_tomatoes_flawless
mode, res = detect_tomatoes_flawless(img)

annotated = img.copy()
for r in res:
    b = [int(x) for x in r["box"]]
    cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    cv2.putText(annotated, r["class_name"], (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite(r"d:\project\TomatoVision\tomatovision-ml\test_annotated_flawless.jpg", annotated)
print(f"Saved test_annotated_flawless.jpg with {len(res)} boxes!")
