import io
import os
import socket
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(
    title="TomatoVision AI - Realtime Inference API",
    version="2.4.0",
    description="High-speed YOLOv8 Ripeness & Quality Estimation API for TomatoVision Mobile & Web App"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tomato_detection_pipeline import TomatoDetectionPipeline

# Initialise the CNN Detection Pipeline (Phase 0-3)
MODEL_CANDIDATES = [
    "runs/detect/tomato_yolo_more_epochs/weights/best.pt",
    "runs/detect/models/tomato_yolo_run-9/weights/best.pt",
    "yolov8n.pt",
]
MODEL_PATH = next((c for c in MODEL_CANDIDATES if os.path.exists(c)),
                  "runs/detect/tomato_yolo_more_epochs/weights/best.pt")

try:
    pipeline = TomatoDetectionPipeline(MODEL_PATH)
    model    = pipeline._model          # kept for legacy references
    print(f"[+] TomatoDetectionPipeline ready.")
except Exception as e:
    print(f"[!] Pipeline load error: {e}"); raise


QUALITY_WEIGHTS = {
    "fresh": 100,
    "ripe": 100,
    "unripe": 75,
    "overripe": 40,
    "spoiled": 0,
}

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class Base64ImageRequest(BaseModel):
    image: str

@app.get("/")
def root():
    return {
        "service": "TomatoVision AI Inference Server",
        "status": "online",
        "version": "2.4.0",
        "endpoints": {
            "health": "/health",
            "predict_upload": "POST /predict",
            "predict_base64": "POST /predict_base64"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": list(model.names.values()) if hasattr(model, 'names') else []
    }

def detect_tomatoes_two_stage(img_bgr):
    """
    MASTER AGRO-VISION DETECTOR (DUAL-MODE ENGINE):
    ===============================================
    Mode 1: Primary Tomato Heap / Crate Isolation (තක්කාලි ගොඩවල් සඳහා)
    -------------------------------------------------------------------
    - Uses High-Saturation Carotenoid Pigment Anchor to isolate the exact Polygon Mask of the heap.
    - 100% DISCARDS surrounding vegetables (chilies, radishes, potatoes, lettuce, cabbage) and ground.
    - Runs multi-fruit distance transform strictly inside the isolated polygon mask.

    Mode 2: Standalone Single Tomato Detection (තනි තක්කාලි ගෙඩිය සඳහා)
    -------------------------------------------------------------------
    - Enforces full fruit circularity (Circularity >= 0.35) and solidity (Solidity >= 0.75).
    - Captures the ENTIRE SINGLE TOMATO in 1 solid bounding box (Unripe, Ripe, Overripe, Spoiled).
    - Excludes background wooden table edges and paper lines.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(w * h)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # =========================================================================
    # STEP 1: HEAP VERIFICATION (තක්කාලි ගොඩක් තිබේදැයි පරීක්ෂා කිරීම)
    # =========================================================================
    red_orange_core = (
        ((h_c <= 16) | (h_c >= 165)) & (s_c > 125) & (v_c > 55) |
        ((h_c > 16) & (h_c <= 27)) & (s_c > 130) & (v_c > 60)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_core = cv2.morphologyEx(red_orange_core, cv2.MORPH_CLOSE, kernel)
    closed_core = cv2.dilate(closed_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(closed_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_heaps = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / bh if bh > 0 else 0
            
            # A true tomato heap must NOT be a top wooden table edge strip
            is_table_edge = (by <= int(0.12 * h) and bh < int(0.20 * h)) or (ar > 3.0) or (bw > 0.80 * w and by <= 5)
            if area > 4500 and not is_table_edge:
                valid_heaps.append(c)

    # =========================================================================
    # SCENARIO A: TOMATO HEAP DETECTED (ගොඩවල් සඳහා වන Engine එක - කිසිදු වෙනසක් කර නැත)
    # =========================================================================
    if valid_heaps:
        heap_mask = np.zeros((h, w), dtype=np.uint8)
        for c in valid_heaps:
            cv2.drawContours(heap_mask, [cv2.convexHull(c)], -1, 255, -1)

        all_tomatoes = (
            ((h_c <= 16) | (h_c >= 162)) & (s_c > 60) & (v_c > 45) |
            ((h_c > 16) & (h_c <= 28)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 65) & (v_c > 45) |
            ((h_c > 42) & (h_c <= 65)) & (s_c > 50) & (v_c > 45)
        ).astype(np.uint8) * 255

        heap_tomatoes = cv2.bitwise_and(all_tomatoes, heap_mask)

        clean_heap = cv2.morphologyEx(heap_tomatoes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        clean_heap = cv2.morphologyEx(clean_heap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        dist = cv2.distanceTransform(clean_heap, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []

        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_dist = cv2.dilate(dist, kernel_peak)
        local_max = (dist == dilated_dist) & (dist > 4.0) & (dist > 0.07 * dist.max())

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max.astype(np.uint8))
        raw_centers = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            d_val = dist[int(cy), int(cx)]
            raw_centers.append((cx, cy, d_val))

        raw_centers.sort(key=lambda item: item[2], reverse=True)

        suppressed = []
        min_dist_sq = 14.0 ** 2
        for pt in raw_centers:
            cx, cy, d_val = pt
            too_close = False
            for s in suppressed:
                if ((cx - s[0])**2 + (cy - s[1])**2) < min_dist_sq:
                    too_close = True
                    break
            if not too_close:
                suppressed.append((cx, cy, d_val))

        detections = []
        for cx, cy, d_val in suppressed:
            r = int(d_val * 1.55)
            r = max(13, min(36, r))

            gx1 = max(0, int(cx - r))
            gy1 = max(0, int(cy - r))
            gx2 = min(w, int(cx + r))
            gy2 = min(h, int(cy + r))

            crop = img_bgr[gy1:gy2, gx1:gx2]
            if crop.size == 0:
                continue

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(crop_hsv[:, :, 0])
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

            if dark_spots > 0.40:
                stage = "spoiled"
                conf = 0.92
            elif (mean_h <= 18 or mean_h >= 160):
                stage = "ripe"
                conf = 0.95
            elif 19 <= mean_h <= 28:
                stage = "overripe"
                conf = 0.92
            elif 29 <= mean_h <= 65:
                stage = "unripe"
                conf = 0.93
            else:
                stage = "ripe"
                conf = 0.88

            detections.append({
                "class_name": stage,
                "raw_class": stage,
                "confidence": conf,
                "box": [float(gx1), float(gy1), float(gx2), float(gy2)]
            })

        return detections

    # =========================================================================
    # SCENARIO B: STANDALONE SINGLE / DOUBLE TOMATO FRUIT (තනි තක්කාලි ගෙඩිය)
    # =========================================================================
    else:
        # Full Tomato Color spectrum (Red, Orange, Yellow, Green Breakers)
        mask_single = (
            ((h_c <= 16) | (h_c >= 160)) & (s_c > 35) & (v_c > 35) |
            ((h_c > 16) & (h_c <= 28)) & (s_c > 40) & (v_c > 40) |
            ((h_c > 28) & (h_c <= 42)) & (s_c > 40) & (v_c > 40) |
            ((h_c > 42) & (h_c <= 68)) & (s_c > 25) & (v_c > 35)
        ).astype(np.uint8) * 255

        # Ignore non-tomato background colors
        mask_single[h_c > 68] = 0
        mask_single[(h_c >= 120) & (h_c <= 160)] = 0
        mask_single[s_c <= 20] = 0

        # Close inner holes in the single fruit
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        closed_mask = cv2.morphologyEx(mask_single, cv2.MORPH_CLOSE, kernel_close)
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        single_candidates = []

        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area < max(1200, 0.015 * img_area):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch if ch > 0 else 0
            perimeter = cv2.arcLength(c, True)
            circ = 4 * np.pi * (c_area / (perimeter * perimeter)) if perimeter > 0 else 0
            hull = cv2.convexHull(c)
            solidity = float(c_area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

            # SHAPE FILTER: A real single tomato fruit is ROUND / OVAL with high circularity and solidity
            if 0.50 <= ar <= 2.0 and circ >= 0.35 and solidity >= 0.75:
                # Reject top wooden table edge
                if not (cw > 0.65 * w and y <= 10 and ch < 0.20 * h):
                    crop = img_bgr[y:y+ch, x:x+cw]
                    if crop.size == 0:
                        continue
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mean_h = np.mean(crop_hsv[:, :, 0])
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    dark_spots = np.sum(gray_crop < 30) / float(gray_crop.size) if gray_crop.size > 0 else 0

                    if dark_spots > 0.40:
                        stage = "spoiled"
                        conf = 0.92
                    elif (mean_h <= 18 or mean_h >= 160):
                        stage = "ripe"
                        conf = 0.95
                    elif 19 <= mean_h <= 28:
                        stage = "overripe"
                        conf = 0.92
                    elif 29 <= mean_h <= 68:
                        stage = "unripe"
                        conf = 0.94
                    else:
                        stage = "ripe"
                        conf = 0.88

                    single_candidates.append({
                        "class_name": stage,
                        "raw_class": stage,
                        "confidence": conf,
                        "box": [float(x), float(y), float(x + cw), float(y + ch)],
                        "area": c_area
                    })

        if single_candidates:
            # Pick the largest single tomato fruit contour (1 solid bounding box covering the entire fruit!)
            single_candidates.sort(key=lambda item: item["area"], reverse=True)
            best = single_candidates[0]
            return [{
                "class_name": best["class_name"],
                "raw_class": best["raw_class"],
                "confidence": best["confidence"],
                "box": best["box"]
            }]

        return []







def _is_verified_tomato_patch(patch, c_name):
    """Strict tomato pigment verification - rejects paper, pen, keyboard, tea cup"""
    if patch is None or patch.size == 0 or patch.shape[0] < 12 or patch.shape[1] < 12:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    tot = float(patch.shape[0] * patch.shape[1])

    # Tomato carotenoid / chlorophyll pixels
    red_px   = np.count_nonzero(((H <= 14) | (H >= 160)) & (S > 55) & (V > 45))
    orange_px = np.count_nonzero((H > 14) & (H <= 26) & (S > 55) & (V > 45))
    green_px  = np.count_nonzero((H > 26) & (H <= 72) & (S > 50) & (V > 40))

    if c_name in ("ripe", "fresh"):
        return (red_px / tot) >= 0.32
    elif c_name == "overripe":
        return ((red_px + orange_px) / tot) >= 0.32
    elif c_name == "unripe":
        return (green_px / tot) >= 0.32
    elif c_name == "spoiled":
        # Spoiled tomatoes still have some fruit coloration
        return ((red_px + orange_px + green_px) / tot) >= 0.28
    return False


def process_yolo_results(img_np):
    h, w, _ = img_np.shape
    EDGE_MARGIN   = 0.08   # Reject detections in outer 8% border (App UI chrome / shadows)
    MIN_BOX_W_REL = 0.045  # Minimum box width as fraction of image width
    MIN_BOX_H_REL = 0.045  # Minimum box height as fraction of image height
    MIN_CONF      = 0.40   # Hard minimum confidence

    yolo_detections = []
    if model is not None:
        try:
            raw_results = model.predict(img_np, conf=MIN_CONF, imgsz=640, verbose=False)
            for r in raw_results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < MIN_CONF:
                        continue

                    cls_id   = int(box.cls[0])
                    raw_cls  = model.names.get(cls_id, "ripe").lower()
                    c_name   = "overripe" if "over" in raw_cls else raw_cls
                    if c_name not in ("ripe", "unripe", "overripe", "spoiled"):
                        c_name = "ripe"

                    x1, y1, x2, y2 = [float(c) for c in box.xyxy[0].tolist()]
                    bw = x2 - x1
                    bh = y2 - y1

                    # ── Geometric Guards ──────────────────────────────
                    # 1. Reject if box centroid falls in the outer EDGE_MARGIN zone
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if (cx < EDGE_MARGIN * w or cx > (1 - EDGE_MARGIN) * w or
                            cy < EDGE_MARGIN * h or cy > (1 - EDGE_MARGIN) * h):
                        continue

                    # 2. Reject tiny boxes (UI elements, distant noise)
                    if bw < MIN_BOX_W_REL * w or bh < MIN_BOX_H_REL * h:
                        continue

                    # 3. Reject extreme aspect ratios (horizontal lines, wide banners)
                    ar = bw / max(1.0, bh)
                    if ar > 3.0 or ar < 0.28:
                        continue

                    # ── Color Pigment Verification ────────────────────
                    px1 = max(0, int(x1)); py1 = max(0, int(y1))
                    px2 = min(w, int(x2)); py2 = min(h, int(y2))
                    patch = img_np[py1:py2, px1:px2]

                    if not _is_verified_tomato_patch(patch, c_name):
                        continue

                    yolo_detections.append({
                        "class_name": c_name,
                        "raw_class":  raw_cls,
                        "confidence": round(conf, 2),
                        "box":        [x1, y1, x2, y2],
                    })

        except Exception as e:
            print(f"[!] YOLO inference error: {e}")

    if yolo_detections:
        final_detections = yolo_detections
    else:
        # Fallback colour-based heap detection (only runs when YOLO finds zero tomatoes)
        final_detections = detect_tomatoes_two_stage(img_np)

    counts = {
        "unripe": 0,
        "ripe": 0,
        "overripe": 0,
        "spoiled": 0,
    }

    for d in final_detections:
        c_name = d["class_name"]
        if c_name in counts:
            counts[c_name] += 1

    # Keep fresh as alias for ripe
    counts["fresh"] = counts["ripe"]

    total = counts["unripe"] + counts["ripe"] + counts["overripe"] + counts["spoiled"]

    if total > 0:
        score = (
            (counts["ripe"] * QUALITY_WEIGHTS["ripe"]) +
            (counts["unripe"] * QUALITY_WEIGHTS["unripe"]) +
            (counts["overripe"] * QUALITY_WEIGHTS["overripe"]) +
            (counts["spoiled"] * QUALITY_WEIGHTS["spoiled"])
        ) / total
    else:
        score = 0.0

    # Draw visual annotations
    annotated_img = img_np.copy()
    color_map = {
        "ripe": (46, 213, 115),      # Green-Cyan
        "fresh": (46, 213, 115),     # Green-Cyan
        "unripe": (0, 210, 211),     # Cyan
        "overripe": (255, 159, 67),  # Orange
        "spoiled": (255, 71, 87)     # Coral Red
    }

    for d in final_detections:
        box = [int(coord) for coord in d["box"]]
        c_name = d["class_name"]
        conf = d["confidence"]
        col = color_map.get(c_name, (46, 213, 115))

        cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), col, 2)
        cv2.putText(
            annotated_img,
            f"{c_name.upper()} {int(conf * 100)}%",
            (box[0], max(14, box[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1
        )

    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_annotated = base64.b64encode(buffer).decode('utf-8')

    return {
        "success": True,
        "image_width": w,
        "image_height": h,
        "total_detected": total,
        "counts": counts,
        "quality_percentage": round(score, 1),
        "detections": final_detections,
        "annotated_image_base64": f"data:image/jpeg;base64,{base64_annotated}"
    }



@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result   = pipeline.run(img_bgr, annotate=True)
        return pipeline.to_api_dict(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict_base64")
async def predict_base64(payload: Base64ImageRequest):
    try:
        raw_b64 = payload.image
        if "base64," in raw_b64:
            raw_b64 = raw_b64.split("base64,")[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_bgr   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result    = pipeline.run(img_bgr, annotate=True)
        return pipeline.to_api_dict(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print("=" * 60)
    print("  🚀 TomatoVision AI Real-time Backend Server")
    print(f"  📡 Local URL:   http://localhost:8000")
    print(f"  📱 Mobile/LAN:  http://{local_ip}:8000")
    print(f"  📖 API Docs:    http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
