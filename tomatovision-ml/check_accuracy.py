from ultralytics import YOLO

# ⚠️ වැදගත්: මෙතන 'train' වෙනුවට ඔයාගේ අලුත්ම Model එක තියෙන ෆෝල්ඩරයේ නම දෙන්න (උදා: train, train2, train3)
model = YOLO("runs/detect/models/tomato_yolo_run-4/weights/best.pt")

print("AI Model එකේ Accuracy එක පරීක්ෂා කරමින් පවතී. කරුණාකර රැඳී සිටින්න...")

# Validation දත්ත හරහා Accuracy එක පරීක්ෂා කිරීම
metrics = model.val()

# Accuracy එක ප්‍රතිශතයක් ලෙස සෑදීම (mAP50)
accuracy = metrics.box.map50
accuracy_percentage = round(accuracy * 100, 2)

print("\n" + "="*45)
print(f"🎉 ඔබගේ AI Model එකේ සාර්ථකත්වය (Accuracy): {accuracy_percentage}%")
print("="*45 + "\n")