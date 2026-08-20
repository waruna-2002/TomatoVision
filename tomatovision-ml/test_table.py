import cv2
import numpy as np

img_path = r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075256134.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Let's crop the actual camera/uploaded image inside the viewfinder or inspect the entire image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

print(f"Image dimensions: {w}x{h}")

# Check the contours in the image
mask_red1 = (h_chan <= 15) & (s_chan > 45) & (v_chan > 40)
mask_red2 = (h_chan >= 162) & (s_chan > 45) & (v_chan > 40)
mask_orange = (h_chan > 15) & (h_chan <= 28) & (s_chan > 55) & (v_chan > 45)
mask_yellow = (h_chan > 28) & (h_chan <= 38) & (s_chan > 55) & (v_chan > 45)
mask_unripe = (h_chan > 38) & (h_chan <= 55) & (s_chan > 45) & (v_chan > 40)

core_mask = (mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_unripe).astype(np.uint8) * 255
core_mask[h_chan > 55] = 0
core_mask[(h_chan >= 120) & (h_chan <= 165)] = 0
core_mask[s_chan <= 40] = 0

kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_clean)

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for idx, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 500:
        continue
    x, y, cw, ch = cv2.boundingRect(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    aspect_ratio = float(cw) / ch if ch > 0 else 0
    extent = area / float(cw * ch) if (cw * ch) > 0 else 0
    
    # Calculate saturation variance and color distribution
    crop_hsv = hsv[y:y+ch, x:x+cw]
    mean_s = np.mean(crop_hsv[:, :, 1])
    mean_h = np.mean(crop_hsv[:, :, 0])
    
    print(f"Contour {idx}: Area={area:.0f} ({(area/(w*h))*100:.1f}%), BBox=[{x}, {y}, {cw}, {ch}], AR={aspect_ratio:.2f}, Circularity={circularity:.2f}, Extent={extent:.2f}, Mean H={mean_h:.1f}, Mean S={mean_s:.1f}")
