import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787102292862.png")[86:528, 412:612]
h, w = img.shape[:2]
img_area = float(w * h)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan = hsv[:, :, 0]
s_chan = hsv[:, :, 1]
v_chan = hsv[:, :, 2]

print(f"H range: [{h_chan.min()}, {h_chan.max()}], Mean: {h_chan.mean():.1f}")
print(f"S range: [{s_chan.min()}, {s_chan.max()}], Mean: {s_chan.mean():.1f}")
print(f"V range: [{v_chan.min()}, {v_chan.max()}], Mean: {v_chan.mean():.1f}")

# Look at the tomato region (center)
center_hsv = hsv[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
print(f"Center H range: [{center_hsv[:,:,0].min()}, {center_hsv[:,:,0].max()}], Median H: {np.median(center_hsv[:,:,0])}")
print(f"Center S range: [{center_hsv[:,:,1].min()}, {center_hsv[:,:,1].max()}], Median S: {np.median(center_hsv[:,:,1])}")
print(f"Center V range: [{center_hsv[:,:,2].min()}, {center_hsv[:,:,2].max()}], Median V: {np.median(center_hsv[:,:,2])}")
