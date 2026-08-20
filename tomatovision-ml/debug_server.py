import cv2
import api_server

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg")
res = api_server.process_yolo_results(img)
print("Direct process_yolo_results result:")
print(f"Total: {res['total_detected']}, Counts: {res['counts']}")
