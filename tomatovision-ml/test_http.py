import requests
import base64

with open(r"C:\Users\Waruna\.gemini\antigravity\brain\d72fea25-187d-4266-99c6-341a7dceced8\.user_uploaded\media_1787105108621.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

r = requests.post("http://localhost:8000/predict_base64", json={"image": b64})
print("HTTP Status:", r.status_code)
data = r.json()
print(f"Total Detected: {data['total_detected']}, Counts: {data['counts']}")
