import cv2
import numpy as np

# Load the 2 market images from user uploads
img1 = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787075584775.jpg") # Market stall with chilies and radishes
img2 = cv2.imread(r"d:\project\TomatoVision\tomatovision-ml\test_images\test_tomatoes.jpg")

# Let us find how to 100% PERFECTLY isolate the Tomato Area:
# A Tomato Pile is characterized by a cluster of RED/ORANGE/YELLOW circular objects.
# Green chilies: Highly elongated (Aspect Ratio > 3.0, thin lines).
# Potatoes: Brown, low saturation (S < 60).
# Sand/Floor: Uniform brown, low saturation (S < 60), 0 circular contours.

def isolate_tomato_cluster(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Tomato Red/Orange/Yellow Carotenoids with high saturation:
    # S > 100 ensures we NEVER touch sand, potatoes, wooden tables, or dirt!
    tomato_pigment = (
        ((h_c <= 14) | (h_c >= 165)) & (s_c > 100) & (v_c > 50) |
        ((h_c > 14) & (h_c <= 32)) & (s_c > 110) & (v_c > 55)
    ).astype(np.uint8) * 255

    # Connect nearby tomatoes in the heap
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    heap_map = cv2.morphologyEx(tomato_pigment, cv2.MORPH_CLOSE, kernel_connect)
    heap_map = cv2.dilate(heap_map, kernel_connect, iterations=2)

    contours, _ = cv2.findContours(heap_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tomato_rois = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 4000:
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw)/ch if ch > 0 else 0
            # Reject horizontal top/bottom background strips
            if ar < 3.2 and not (cw > 0.75 * w and (y <= 5 or ch < 0.12 * h)):
                tomato_rois.append((x, y, cw, ch, area))
                
    return tomato_rois

print("Tomato ROIs in Image 1 (Market Stall):", isolate_tomato_cluster(img1))
