"""
================================================================================
  TomatoVision - CNN-Based Tomato Detection and Counting Pipeline
================================================================================
  Model   : YOLOv8 (CNN backbone: CSPDarknet53 + PANet + Detection Head)
  Dataset : Roboflow - 4 Classes: ['overripe', 'ripe', 'spoiled', 'unripe']
  mAP50   : 81.0%  |  Ripe: 93.7%  |  Spoiled: 90.7%  |  Unripe: 86.0%

  Pipeline:
    Phase 0  - Scene Analyser      (single fruit vs dense group/crate/heap)
    Phase 1  - Single Fruit CNN    (YOLOv8 direct with strict validation)
    Phase 2  - Dense Group CNN     (YOLOv8 + Edge-Separated Distance Peaks)
    Phase 3  - Counter/Classifier  (count, quality score, grade category)
================================================================================
"""

import os, cv2, numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from scipy.ndimage import maximum_filter
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH       = Path(__file__).parent / "runs/detect/tomato_yolo_more_epochs/weights/best.pt"
CONF_THRESHOLD   = 0.40
MIN_BOX_W_REL    = 0.045
MIN_BOX_H_REL    = 0.045
EDGE_MARGIN      = 0.08
PIGMENT_RATIO    = 0.32
QUALITY_WEIGHTS  = {"ripe": 100, "unripe": 75, "overripe": 40, "spoiled": 0}
COLOUR_MAP       = {
    "ripe":     (46,  213, 115),
    "unripe":   (0,   210, 211),
    "overripe": (255, 159,  67),
    "spoiled":  (255,  71,  87),
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FruitDetection:
    class_name: str
    confidence: float
    box:        List[float]
    source:     str = "yolo"

@dataclass
class DetectionResult:
    success:         bool
    total_count:     int
    counts:          Dict[str, int]
    quality_score:   float
    category:        str
    detections:      List[FruitDetection]
    scene_mode:      str
    annotated_image: Optional[np.ndarray] = None

    @property
    def grade(self) -> str:
        s = self.quality_score
        if s >= 90: return "Grade A+ - Premium Export"
        if s >= 75: return "Grade A  - Fresh Market"
        if s >= 55: return "Grade B  - Standard Market"
        if s >= 30: return "Grade C  - Processing"
        return "Reject / Cull"

# ---------------------------------------------------------------------------
# Phase 0  - Scene Analyser
# ---------------------------------------------------------------------------
class SceneAnalyser:
    @staticmethod
    def _tomato_mask(img_320):
        hsv = cv2.cvtColor(img_320, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        return ((((H<=14)|(H>=160))&(S>55)&(V>45)) |
                ((H>14)&(H<=28)&(S>55)&(V>45)) |
                ((H>28)&(H<=72)&(S>50)&(V>40))).astype(np.uint8)*255

    @classmethod
    def analyse(cls, img_bgr) -> Tuple[str, float]:
        small   = cv2.resize(img_bgr, (320, 320))
        mask    = cls._tomato_mask(small)
        density = float(np.count_nonzero(mask)) / (320.0 * 320.0)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return "single_fruit", 0.0

        rw = (xs.max() - xs.min()) / 320.0
        rh = (ys.max() - ys.min()) / 320.0
        area_frac = rw * rh

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 100)
        internal_edges = cv2.bitwise_and(edges, edges, mask=mask)
        edge_density = float(np.count_nonzero(internal_edges)) / float(max(1, np.count_nonzero(mask)))

        is_dense = (density > 0.18) and (area_frac > 0.40) and (edge_density > 0.12)
        mode = "dense_group" if is_dense else "single_fruit"
        return mode, density

# ---------------------------------------------------------------------------
# Phase 1  - Single Fruit CNN Detector (YOLOv8)
# ---------------------------------------------------------------------------
class SingleFruitDetector:
    def __init__(self, model: YOLO):
        self._model = model

    @staticmethod
    def _verify_patch(patch, c_name) -> bool:
        if patch is None or patch.size==0 or patch.shape[0]<12 or patch.shape[1]<12:
            return False
        hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H,S,V= hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        tot  = float(patch.shape[0]*patch.shape[1])
        r  = np.count_nonzero(((H<=14)|(H>=160))&(S>55)&(V>45))
        o  = np.count_nonzero((H>14)&(H<=26)&(S>55)&(V>45))
        g  = np.count_nonzero((H>26)&(H<=72)&(S>50)&(V>40))
        if c_name=="ripe":     return (r/tot)       >= PIGMENT_RATIO
        if c_name=="overripe": return ((r+o)/tot)   >= PIGMENT_RATIO
        if c_name=="unripe":   return (g/tot)        >= PIGMENT_RATIO
        if c_name=="spoiled":  return ((r+o+g)/tot) >= PIGMENT_RATIO*0.85
        return False

    def detect(self, img_bgr) -> List[FruitDetection]:
        h, w = img_bgr.shape[:2]
        dets = []
        for r in self._model.predict(img_bgr, conf=CONF_THRESHOLD, imgsz=640, verbose=False):
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD: continue
                raw  = self._model.names.get(int(box.cls[0]), "ripe").lower()
                cn   = "overripe" if "over" in raw else raw
                if cn not in ("ripe","unripe","overripe","spoiled"): cn="ripe"
                x1,y1,x2,y2 = [float(c) for c in box.xyxy[0].tolist()]
                bw,bh = x2-x1, y2-y1
                cx,cy = (x1+x2)/2, (y1+y2)/2
                if cx<EDGE_MARGIN*w or cx>(1-EDGE_MARGIN)*w: continue
                if cy<EDGE_MARGIN*h or cy>(1-EDGE_MARGIN)*h: continue
                if bw<MIN_BOX_W_REL*w or bh<MIN_BOX_H_REL*h: continue
                ar = bw/max(1,bh)
                if ar>3.0 or ar<0.28: continue
                patch = img_bgr[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))]
                if not self._verify_patch(patch, cn): continue
                dets.append(FruitDetection(cn, round(conf,3), [x1,y1,x2,y2], "yolo"))
        return dets

# ---------------------------------------------------------------------------
# Phase 2  - Dense Group CNN + Distance Transform Peak Segmentation
# ---------------------------------------------------------------------------
class DenseGroupDetector:
    def __init__(self, model, sd: SingleFruitDetector):
        self._model = model
        self._sd    = sd

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter   = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def detect(self, img_bgr) -> List[FruitDetection]:
        h, w = img_bgr.shape[:2]
        yolo_dets = self._sd.detect(img_bgr)

        # Tomato mask
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        mask = ((((H<=14)|(H>=160))&(S>55)&(V>45)) |
                ((H>14)&(H<=28)&(S>55)&(V>45)) |
                ((H>28)&(H<=72)&(S>50)&(V>40))).astype(np.uint8)*255

        # Ignore outer margin
        margin_y = int(h * 0.10); margin_x = int(w * 0.10)
        mask[:margin_y, :] = 0; mask[-margin_y:, :] = 0
        mask[:, :margin_x] = 0; mask[:, -margin_x:] = 0

        # Canny edges to split touching fruits
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 35, 110)
        k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.dilate(edges, k_edge, iterations=1)

        separated_mask = cv2.subtract(mask, dilated_edges)
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        separated_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, k_clean, iterations=1)

        dist = cv2.distanceTransform(separated_mask, cv2.DIST_L2, 5)
        dist_smooth = cv2.GaussianBlur(dist, (7, 7), 0)

        window_size = max(23, int(min(h, w) / 16))
        if window_size % 2 == 0: window_size += 1

        local_max = maximum_filter(dist_smooth, size=window_size) == dist_smooth
        peaks = local_max & (dist_smooth > 4.5)

        ys, xs = np.where(peaks)
        peak_dets = []
        avg_r = max(16, int(min(h, w) / 24))
        for x, y in zip(xs, ys):
            val = dist_smooth[y, x]
            r = int(max(avg_r * 0.75, min(avg_r * 1.25, val * 1.3)))
            x1 = max(0.0, float(x - r)); y1 = max(0.0, float(y - r))
            x2 = min(float(w), float(x + r)); y2 = min(float(h), float(y + r))

            patch = img_bgr[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0: continue
            phsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            pH = phsv[:,:,0].flatten()
            pS = phsv[:,:,1].flatten()
            pV = phsv[:,:,2].flatten()
            pH = pH[(pS > 45) & (pV > 40)]
            if len(pH) == 0: continue
            mh = float(np.mean(pH))
            if mh <= 16 or mh >= 158: c_name = "ripe"
            elif 16 < mh <= 28: c_name = "overripe"
            elif 28 < mh <= 72: c_name = "unripe"
            else: c_name = "spoiled"

            peak_dets.append(FruitDetection(c_name, 0.88, [x1, y1, x2, y2], "dense_peak"))

        # Merge YOLO and peak detections via IoU NMS
        final_dets = list(yolo_dets)
        for pd in peak_dets:
            if not any(self._iou(pd.box, yd.box) > 0.35 for yd in yolo_dets):
                final_dets.append(pd)

        return final_dets if len(final_dets) >= len(yolo_dets) else yolo_dets

# ---------------------------------------------------------------------------
# Phase 3  - Counter & Classifier
# ---------------------------------------------------------------------------
class FruitCounter:
    @staticmethod
    def count(dets):
        c = {"ripe":0,"unripe":0,"overripe":0,"spoiled":0}
        for d in dets:
            if d.class_name in c: c[d.class_name]+=1
        return c

    @staticmethod
    def quality_score(counts):
        total = sum(counts.values())
        if total==0: return 0.0
        return round(sum(counts.get(k,0)*w for k,w in QUALITY_WEIGHTS.items())/total, 1)

    @staticmethod
    def categorise(n):
        if n<=0:   return "None Detected"
        if n==1:   return "Single Fruit"
        if n<=5:   return "Few (2-5)"
        if n<=15:  return "Moderate Group (6-15)"
        if n<=50:  return "Large Group (16-50)"
        return "Very Large Group (50+)"

# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------
def annotate_image(img, dets, result):
    out = img.copy()
    h,w = out.shape[:2]
    for d in dets:
        if len(d.box)<4: continue
        x1,y1,x2,y2 = [int(c) for c in d.box]
        col = COLOUR_MAP.get(d.class_name,(46,213,115))
        lw  = 1 if d.source=="dense_peak" else 2
        cv2.rectangle(out,(x1,y1),(x2,y2),col,lw)
        lbl = f"{d.class_name.upper()} {int(d.confidence*100)}%"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
        ty = max(th+2,y1-3)
        cv2.rectangle(out,(x1,ty-th-2),(x1+tw+4,ty+2),col,-1)
        cv2.putText(out,lbl,(x1+2,ty),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    lines=[f"Scene: {result.scene_mode.replace('_',' ').title()}",
           f"Total: {result.total_count}  ({result.category})",
           f"Ripe: {result.counts.get('ripe',0)}  Unripe: {result.counts.get('unripe',0)}",
           f"Overripe: {result.counts.get('overripe',0)}  Spoiled: {result.counts.get('spoiled',0)}",
           f"Quality: {result.quality_score:.1f}%  |  {result.grade}"]
    bh = len(lines)*20+20
    cv2.rectangle(out,(0,h-bh),(w,h),(10,14,30),-1)
    for i,l in enumerate(lines):
        cv2.putText(out,l,(10,h-bh+20+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,230,255),1,cv2.LINE_AA)
    return out

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class TomatoDetectionPipeline:
    def __init__(self, model_path=str(MODEL_PATH)):
        print(f"[TomatoVision] Loading CNN model: {model_path}")
        self._model   = YOLO(model_path)
        self._sd      = SingleFruitDetector(self._model)
        self._dgd     = DenseGroupDetector(self._model, self._sd)
        self._counter = FruitCounter()
        print(f"[TomatoVision] Ready. Classes: {self._model.names}")

    def run(self, img_bgr, annotate=True) -> DetectionResult:
        if img_bgr is None or img_bgr.size==0:
            return DetectionResult(False,0,{},0.0,"None Detected",[],  "unknown")

        # Phase 0 - Scene Analysis
        scene_mode, _ = SceneAnalyser.analyse(img_bgr)

        # Phase 1 / 2 - CNN Detection
        dets = (self._dgd.detect(img_bgr) if scene_mode=="dense_group"
                else self._sd.detect(img_bgr))

        # Phase 3 - Count & Classify
        counts = self._counter.count(dets)
        total  = sum(counts.values())
        score  = self._counter.quality_score(counts)
        cat    = self._counter.categorise(total)

        result = DetectionResult(True, total, counts, score, cat, dets, scene_mode)
        if annotate:
            result.annotated_image = annotate_image(img_bgr, dets, result)
        return result

    def run_file(self, path, annotate=True) -> DetectionResult:
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(path)
        return self.run(img, annotate)

    def run_bytes(self, image_bytes, annotate=True) -> DetectionResult:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return self.run(cv2.imdecode(arr, cv2.IMREAD_COLOR), annotate)

    def to_api_dict(self, result: DetectionResult) -> dict:
        import base64
        b64 = None
        if result.annotated_image is not None:
            _, buf = cv2.imencode(".jpg", result.annotated_image)
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        return {
            "success":                result.success,
            "total_detected":         result.total_count,
            "counts": {**result.counts, "fresh": result.counts.get("ripe",0)},
            "quality_percentage":     result.quality_score,
            "grade":                  result.grade,
            "category":               result.category,
            "scene_mode":             result.scene_mode,
            "detections":             [{"class_name":d.class_name,"confidence":d.confidence,
                                        "box":d.box,"source":d.source} for d in result.detections],
            "annotated_image_base64": b64,
        }

if __name__ == "__main__":
    import sys, json, argparse
    p = argparse.ArgumentParser(description="TomatoVision CNN Pipeline")
    s = p.add_subparsers(dest="cmd")
    d = s.add_parser("detect");  d.add_argument("image"); d.add_argument("--save"); d.add_argument("--json", action="store_true"); d.add_argument("--model", default=str(MODEL_PATH))
    args = p.parse_args()

    if args.cmd == "detect":
        pl = TomatoDetectionPipeline(args.model)
        r  = pl.run_file(args.image)
        print(f"\nScene    : {r.scene_mode.replace('_',' ').title()}")
        print(f"Total    : {r.total_count}  [{r.category}]")
        print(f"Counts   : {r.counts}")
        print(f"Quality  : {r.quality_score:.1f}%  |  {r.grade}\n")
