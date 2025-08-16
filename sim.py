from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import pyttsx3
import time
from datetime import datetime
from fpdf import FPDF
import os

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
#change

# Video Capture
cap = cv2.VideoCapture(0)
model = YOLO("skinmodel.pt")

# Detection parameters
CONFIDENCE_THRESHOLD = 0.7  # Only consider detections >70% confidence
NMS_THRESHOLD = 0.5  # Non-Maximum Suppression threshold
ANNOUNCE_COOLDOWN = 5  # seconds between announcements

# Enhanced disease database
disease_info = {
    "Acne": {
        "description": "A skin condition causing pimples, blackheads, and inflammation due to clogged hair follicles and oil buildup.",
        "precaution": "Keep skin clean, avoid oily products, don't squeeze pimples.",
        "medicine": "Benzoyl peroxide or salicylic acid treatments, or consult a dermatologist for prescription creams."
    },
    "Chickenpox": {
        "description": "A highly contagious viral infection causing itchy, blister-like rashes all over the body.",
        "precaution": "Isolate the patient, avoid scratching to prevent infection and scarring.",
        "medicine": "Calamine lotion for itching and antiviral drugs like Acyclovir under medical supervision."
    },
    "Eczema": {
        "description": "A condition that makes skin red, itchy, cracked, and inflamed, often due to an overactive immune response.",
        "precaution": "Moisturize regularly, avoid harsh soaps and irritants.",
        "medicine": "Topical corticosteroids like hydrocortisone and prescription ointments."
    },
    "Monkeypox": {
        "description": "A viral disease causing fever, swollen lymph nodes, and a blistering rash, spread through close contact.",
        "precaution": "Isolate the patient, avoid physical contact, maintain hygiene.",
        "medicine": "Antiviral treatment like Tecovirimat under a doctor's guidance."
    },
    "Psoriasis": {
        "description": "A chronic autoimmune skin disease that speeds up skin cell production, causing scaly, red patches.",
        "precaution": "Keep skin moisturized, avoid skin injuries, manage stress.",
        "medicine": "Topical corticosteroids, vitamin D analogs, and phototherapy as prescribed."
    },
    "Ringworm": {
        "description": "A contagious fungal infection causing circular, red, scaly rashes on the skin or scalp.",
        "precaution": "Keep the infected area dry and clean, avoid sharing towels or clothes.",
        "medicine": "Over-the-counter antifungal creams like Clotrimazole or Terbinafine."
    },
    "basal cell carcinoma": {
        "description": "A type of skin cancer that appears as a waxy bump, usually caused by long-term sun exposure.",
        "precaution": "Limit sun exposure, use sunscreen, wear protective clothing.",
        "medicine": "Requires medical consultation for surgery, topical medications, or radiation therapy."
    },
    "melanoma": {
        "description": "A serious form of skin cancer that develops in the pigment-producing cells, often appearing as a new or changing mole.",
        "precaution": "Avoid excessive UV exposure, perform regular skin self-checks.",
        "medicine": "Immediate medical intervention is crucial — treatment may include surgery, immunotherapy, or targeted therapy depending on stage."
    },
    "tinea-versicolor": {
        "description": "A fungal infection that causes discolored patches on the skin, often on the chest, back, and upper arms.",
        "precaution": "Keep the skin dry, avoid excessive sweating, wear breathable clothing.",
        "medicine": "Antifungal shampoos and topical creams like Ketoconazole or Selenium sulfide."
    },
    "vitiligo": {
        "description": "A condition where the skin loses melanin, resulting in white patches on different parts of the body.",
        "precaution": "Protect affected skin with sunscreen, avoid skin injuries or friction.",
        "medicine": "Topical corticosteroids, UVB light therapy, or consult a dermatologist for advanced treatments."
    },
    "warts": {
        "description": "Small, grainy skin growths caused by the human papillomavirus (HPV) that are often rough to the touch.",
        "precaution": "Avoid scratching or picking, maintain personal hygiene, avoid sharing personal items.",
        "medicine": "Over-the-counter salicylic acid treatments or doctor-administered freezing (cryotherapy)."
    }
}

classNames = ["Acne", "Chickenpox", "Eczema", "Monkeypox", "Psoriasis",
              "Ringworm", "basal cell carcinoma", "melanoma",
              "tinea-versicolor", "vitiligo", "warts"]

# Variables to control announcements
last_detected = ""
last_announce_time = 0


def apply_nms(boxes, scores, threshold):
    """Apply Non-Maximum Suppression to eliminate overlapping boxes"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)

    keep = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last],
                                                     np.where(overlap > threshold)[0])))

    return keep


def create_pdf_report(disease_name, confidence, img):
    """Generate PDF report for detected disease"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Skin Disease Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Disease info
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Detected: {disease_name} (Confidence: {confidence * 100:.1f}%)", ln=True)
    pdf.ln(5)

    # Save and add image
    img_path = f"temp_{disease_name}.jpg"
    cv2.imwrite(img_path, img)
    pdf.image(img_path, x=10, w=180)
    pdf.ln(5)

    # Add disease details
    if disease_name in disease_info:
        info = disease_info[disease_name]
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Description:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, info["description"])

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Precautions:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, info["precaution"])

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Recommended Treatment:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, info["medicine"])

    # Save PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"Disease_Report_{disease_name}_{timestamp}.pdf"
    pdf.output(pdf_filename)
    os.remove(img_path)

    return pdf_filename


def speak(text):
    """Convert text to speech"""
    print(f"ANNOUNCEMENT: {text}")
    engine.say(text)
    engine.runAndWait()


def get_disease_info(disease_name):
    """Get complete information about a disease"""
    if disease_name in disease_info:
        info = disease_info[disease_name]
        return (f"Detected {disease_name}. Description: {info['description']} "
                f"Precautions: {info['precaution']} "
                f"Recommended treatment: {info['medicine']}")
    return f"{disease_name} detected. Please consult a dermatologist."


def process_detections(img):
    """Process detections and return best detection"""
    results = model(img, stream=True, conf=CONFIDENCE_THRESHOLD)

    boxes = []
    confidences = []
    class_ids = []

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_ids.append(cls)

    indices = apply_nms(boxes, confidences, NMS_THRESHOLD)

    if len(indices) > 0:
        best_idx = indices[0]
        return {
            'box': boxes[best_idx],
            'conf': confidences[best_idx],
            'class': classNames[class_ids[best_idx]]
        }
    return None


# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    current_time = time.time()

    # Process detections
    detection = process_detections(img)

    if detection:
        x1, y1, x2, y2 = detection['box']
        conf = detection['conf']
        current_class = detection['class']

        # Visual feedback
        cvzone.putTextRect(img, f'{current_class} {conf:.2f}',
                           (max(0, x1), max(35, y1)),
                           scale=1, thickness=1,
                           colorB=(0, 0, 255), colorT=(255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Announcement logic
        if current_class != last_detected and (current_time - last_announce_time) > ANNOUNCE_COOLDOWN:
            full_info = get_disease_info(current_class)
            speak(full_info)
            last_detected = current_class
            last_announce_time = current_time

    # Show instructions
    cvzone.putTextRect(img, "SPACE: Save Report | Q: Quit",
                       (20, 20), scale=1, thickness=1,
                       colorB=(0, 0, 0), colorT=(255, 255, 255))

    cv2.imshow("Skin Disease Detector", img)

    # Key controls
    key = cv2.waitKey(1)
    if key == 32:  # SPACEBAR
        if last_detected and detection and detection['class'] == last_detected:
            # Draw green box
            x1, y1, x2, y2 = detection['box']
            cvzone.putTextRect(img, f'SAVED: {last_detected}',
                               (max(0, x1), max(35, y1)),
                               scale=1, thickness=2,
                               colorB=(0, 255, 0), colorT=(0, 0, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Generate report
            pdf_file = create_pdf_report(last_detected, detection['conf'], img)
            speak(f"Report saved as {pdf_file}")

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()