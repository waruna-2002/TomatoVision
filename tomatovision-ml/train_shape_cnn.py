from ultralytics import YOLO
from pathlib import Path

def train_shape_model():
    # 1. කලින් Train වූ හොඳම weights තිබේදැයි බැලීම (නැතිනම් yolov8n.pt ගනී)
    pretrained_weights = "runs/detect/models/tomato_yolo_run-8/weights/best.pt"
    
    if Path(pretrained_weights).exists():
        print(f"\n[+] කලින් Train වූ Best Weights හමු විය: {pretrained_weights}")
        print("[+] එම Weights මත පදනම්ව Shape Detection සඳහා අලුතින් Fine-Tuning ආරම්භ වේ...\n")
        model = YOLO(pretrained_weights)
    else:
        print("\n[*] මූලික YOLOv8n Weights මඟින් Shape Training ආරම්භ වේ...\n")
        model = YOLO("yolov8n.pt")

    # 2. Shape-focused Augmentations සමඟ සම්පූර්ණ Training එක ක්‍රියාත්මක කිරීම
    results = model.train(
        data="dataset/data.yaml",       # Dataset Path
        epochs=50,                      # තවත් අමතර Epochs 50ක් සම්පූර්ණයෙන්ම Train වේ
        imgsz=640,
        batch=16,
        workers=2,
        project="runs/shape_detect",
        name="tomato_shape_perfect",
        exist_ok=True,
        
        # --- Shape & Spatial Geometry Augmentations (හැඩය ඉගෙන ගැනීමට) ---
        degrees=15.0,                   # කෝණ වෙනස්වීම් (Rotation)
        scale=0.5,                      # කුඩා/විශාල ප්‍රමාණ (Scaling)
        shear=2.5,                      # Perspective Shearing
        perspective=0.0005,             # Camera Tilt
        fliplr=0.5,                     # Horizontal Flip
        flipud=0.2,                     # Vertical Flip
        mosaic=1.0,                     # Clusters / Bunches හඳුනාගැනීමට
        mixup=0.15,                     # එක මත එක ඇති තක්කාලි වෙන් කරගැනීමට
        
        # --- Color Bias අවම කිරීම ---
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        
        optimizer="AdamW",
        lr0=0.0008                      # Fine-tuning සඳහා ප්‍රශස්ත Learning Rate එකක්
    )

    print("\n✅ Shape Detection Training එක සාර්ථකව අවසන් විය!")
    print("📁 නව Model එක Save වූ ස්ථානය: runs/shape_detect/tomato_shape_perfect/weights/best.pt")

if __name__ == '__main__':
    train_shape_model()