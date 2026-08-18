from ultralytics import YOLO

def main():
    # Model එක ලබාදීම
    model = YOLO("yolov8n.pt") 

    # වැඩි වට ගණනක් (epochs) යොදා Train කිරීම
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,  # <--- මෙන්න මෙතනින් තමයි වට ගණන වැඩි කරලා තියෙන්නේ
        imgsz=640,
        batch=16,
        name="tomato_yolo_more_epochs" # අලුත් ෆෝල්ඩරයක සේව් වෙන්න නමක්
    )

if __name__ == '__main__':
    main()