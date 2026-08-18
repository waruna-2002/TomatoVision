import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from fpdf import FPDF

# 1. Define Weights and Parameters
WEIGHTS = {
    "ripe": 100,
    "unripe": 75,
    "overripe": 40,
    "spoiled": 0
}

def analyze_image(image_path, model_path="runs/detect/models/tomato_yolo_run-2/weights/best.pt"):
    """Scan the image using the YOLO model and extract data."""
    # Load the YOLO model (provide your newly trained model path here)
    model = YOLO(model_path) 
    
    # Scan the image
    results = model(image_path)[0]
    
    # Dictionary to count detected tomatoes
    counts = {"ripe": 0, "unripe": 0, "overripe": 0, "spoiled": 0}
    
    # Get labels of the detected tomatoes
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower()
        if class_name in counts:
            counts[class_name] += 1
            
    # Save the new image with bounding boxes
    output_img_path = "scanned_result.jpg"
    annotated_frame = results.plot()
    cv2.imwrite(output_img_path, annotated_frame)
    
    return counts, output_img_path

def calculate_quality(counts):
    """Calculate the overall quality percentage of the tomatoes."""
    total_tomatoes = sum(counts.values())
    
    if total_tomatoes == 0:
        return 0, 0
        
    total_score = (
        (counts["ripe"] * WEIGHTS["ripe"]) +
        (counts["unripe"] * WEIGHTS["unripe"]) +
        (counts["overripe"] * WEIGHTS["overripe"]) +
        (counts["spoiled"] * WEIGHTS["spoiled"])
    )
    
    quality_percentage = total_score / total_tomatoes
    return quality_percentage, total_tomatoes

def generate_pdf_report(counts, quality_percentage, total_tomatoes, img_path, output_pdf="Tomato_Analysis_Report.pdf"):
    """Generate the analysis report as a PDF."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="TomatoVision - Batch Analysis Report", ln=True, align='C')
    
    # Date and Time
    pdf.set_font("Arial", size=12)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Scan Date & Time: {now}", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    
    # Summary
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Overall Batch Quality: {quality_percentage:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Total Tomatoes Detected: {total_tomatoes}", ln=True)
    
    # Classification
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Details by Category:", ln=True)
    for category, count in counts.items():
        pdf.cell(200, 10, txt=f"  - {category.capitalize()}: {count}", ln=True)
        
    # Add image to the PDF
    pdf.ln(10)
    pdf.cell(200, 10, txt="Scanned Image Preview:", ln=True)
    pdf.image(img_path, x=20, w=150)
    
    # Save the PDF
    pdf.output(output_pdf)
    print(f"\nReport successfully saved as: {output_pdf}")

# --- Main Execution ---
if __name__ == "__main__":
    IMAGE_TO_SCAN = "test_images/test_tomatoes.jpg" # Replace with your image file name
    
    print("Scanning image...")
    tomato_counts, result_image = analyze_image(IMAGE_TO_SCAN)
    
    print("Calculating quality...")
    quality, total = calculate_quality(tomato_counts)
    
    print("Generating PDF report...")
    generate_pdf_report(tomato_counts, quality, total, result_image)