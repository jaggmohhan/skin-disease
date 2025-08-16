from ultralytics import YOLO
import cv2
import cvzone
import math
import pyttsx3
import time
from datetime import datetime

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking speed

# Video Capture
cap = cv2.VideoCapture(0)
model = YOLO("skinmodel.pt")

# Disease database
disease_info = {
    "Acne": {
        "precaution": "Keep skin clean, avoid oily products, don't squeeze pimples.",
        "medicine": "Use benzoyl peroxide or salicylic acid treatments."
    },
    "Ringworm": {
        "precaution": "Keep area dry, avoid sharing personal items, wash bedding regularly.",
        "medicine": "Apply antifungal creams like clotrimazole or terbinafine."
    },
    "warts": {
        "precaution": "Don't pick at warts, keep hands clean, avoid sharing towels.",
        "medicine": "Salicylic acid patches or doctor-administered cryotherapy."
    },
    # Add more diseases as needed
}

classNames = ["Acne", "Chickenpox", "Eczema", "Monkeypox", "Psoriasis",
              "Ringworm", "basal cell carcinoma", "melanoma",
              "tinea-versicolor", "vitiligo", "warts"]


def speak(text):
    print(f"ANNOUNCEMENT: {text}")  # Debug output
    engine.say(text)
    engine.runAndWait()


def analyze_and_announce(img):
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil(box.conf[0].item() * 100) / 100
            if conf > 0.5:  # Only consider confident detections
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in disease_info:
                    info = disease_info[currentClass]
                    message = (f"Detected {currentClass} with {conf * 100}% confidence. "
                               f"Precautions: {info['precaution']} "
                               f"Recommended treatment: {info['medicine']}")

                    # Visual feedback
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cvzone.putTextRect(img, f'ANALYZED: {currentClass}',
                                       (max(0, x1), max(35, y1)),
                                       scale=1, thickness=2,
                                       colorB=(0, 255, 0), colorT=(0, 0, 0))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Save the analyzed image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"analysis_{currentClass}_{timestamp}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"Saved analysis as {filename}")

                    return message
    return "No disease detected with sufficient confidence."


while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    # Display live detection (without analysis)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0].item() * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1,
                                   colorB=(0, 0, 255), colorT=(255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Skin Disease Detector (Press SPACE to analyze)", img)

    key = cv2.waitKey(1)
    if key == 32:  # SPACEBAR key code
        message = analyze_and_announce(img.copy())  # Analyze a copy of current frame
        speak(message)
        cv2.imshow("Last Analysis", img)  # Show the analyzed frame

    elif key == ord('q'):  # Quit on 'q'
        break

cap.release()
cv2.destroyAllWindows()