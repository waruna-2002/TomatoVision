import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Actual Tomato Pile is around X: [260, 550], Y: [210, 600]
tomato_crop = hsv[250:500, 300:500]
print(f"Actual Tomatoes HSV -> H: [{tomato_crop[:,:,0].min()}, {tomato_crop[:,:,0].max()}], mean H={tomato_crop[:,:,0].mean():.1f}, S: mean={tomato_crop[:,:,1].mean():.1f} (min={tomato_crop[:,:,1].min()}, max={tomato_crop[:,:,1].max()}), V: mean={tomato_crop[:,:,2].mean():.1f}")

# Sandy Ground at bottom is Y: [700, 950], X: [100, 600]
ground_crop = hsv[750:900, 200:500]
print(f"Ground Floor HSV -> H: [{ground_crop[:,:,0].min()}, {ground_crop[:,:,0].max()}], mean H={ground_crop[:,:,0].mean():.1f}, S: mean={ground_crop[:,:,1].mean():.1f} (min={ground_crop[:,:,1].min()}, max={ground_crop[:,:,1].max()}), V: mean={ground_crop[:,:,2].mean():.1f}")

# Green Chilies on the left is X: [50, 300], Y: [200, 600]
chili_crop = hsv[250:500, 50:250]
print(f"Green Chilies HSV -> H: [{chili_crop[:,:,0].min()}, {chili_crop[:,:,0].max()}], mean H={chili_crop[:,:,0].mean():.1f}, S: mean={chili_crop[:,:,1].mean():.1f}")
