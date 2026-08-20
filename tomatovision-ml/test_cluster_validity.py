import cv2
import numpy as np

def is_valid_tomato_cluster(rx, ry, rw, rh, w, h, contour_area):
    ar = float(rw) / rh if rh > 0 else 0
    # Reject flat horizontal table borders
    if ar > 2.3 or ar < 0.40:
        return False
    # Reject edge strips touching top/bottom with thin height
    if rw > 0.60 * w and rh < 0.20 * h:
        return False
    if contour_area < 2500:
        return False
    return True

print("Table strip AR test:")
print("  Table strip 362x96 (W=534, H=1001) valid? ", is_valid_tomato_cluster(95, 84, 362, 96, 534, 1001, 26000))
print("  Market Heap 397x484 (W=768, H=1024) valid? ", is_valid_tomato_cluster(219, 179, 397, 484, 768, 1024, 45000))
print("  Crate Heap 450x550 (W=534, H=700) valid? ", is_valid_tomato_cluster(50, 60, 450, 550, 534, 700, 80000))
