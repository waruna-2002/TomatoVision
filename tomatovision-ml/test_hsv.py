import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tomato region: [300:500, 250:450]
tomato_roi = hsv[300:500, 250:450]
# Sand floor region: [800:1000, 250:450]
sand_roi = hsv[800:1000, 250:450]
# Lettuce region: [300:500, 20:150]
lettuce_roi = hsv[300:500, 20:150]
# Potato region: [300:500, 600:750]
potato_roi = hsv[300:500, 600:750]

print("Tomato Region -> Mean H:", np.mean(tomato_roi[:,:,0]), "Mean S:", np.mean(tomato_roi[:,:,1]), "Mean V:", np.mean(tomato_roi[:,:,2]))
print("Sand Floor    -> Mean H:", np.mean(sand_roi[:,:,0]), "Mean S:", np.mean(sand_roi[:,:,1]), "Mean V:", np.mean(sand_roi[:,:,2]))
print("Lettuce       -> Mean H:", np.mean(lettuce_roi[:,:,0]), "Mean S:", np.mean(lettuce_roi[:,:,1]), "Mean V:", np.mean(lettuce_roi[:,:,2]))
print("Potatoes      -> Mean H:", np.mean(potato_roi[:,:,0]), "Mean S:", np.mean(potato_roi[:,:,1]), "Mean V:", np.mean(potato_roi[:,:,2]))
