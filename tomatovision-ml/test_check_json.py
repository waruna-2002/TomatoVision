import requests
import base64
import cv2

img = cv2.imread(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787106806236.png")
viewport = img[88:544, 403:597]

_, buf = cv2.imencode(".jpg", viewport)
b64 = base64.b64encode(buf).decode("utf-8")

r = requests.post("http://localhost:8000/predict_base64", json={"image": b64})
data = r.json()
print("Total Detected:", data.get("total_detected"))
print("Counts:", data.get("counts"))
print("Detections:", data.get("detections"))
