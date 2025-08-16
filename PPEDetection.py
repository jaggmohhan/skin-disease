from ultralytics import YOLO
import cv2
import cvzone
import math
import pygame

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = "alertForWeapon.mp3"  # Ensure this file is in the correct directory

# Load the alert sound
pygame.mixer.music.load(alert_sound)

# Video Capture
cap = cv2.VideoCapture(0)  # Use 1 for webcam, 0 for video file

model = YOLO("best_weapons_model.pt")

classNames = ["axe", "bomb", "bow", "cleaver", "cutlass", "katana", "knife", "mace",
              "machine gun", "morningstar", "pistol", "rifle", "rocket launcher",
              "scabbard", "scope", "shield", "shotgun", "sickle", "smg", "sniper rifle",
              "spear", "sword", "war hammer"]

myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            if conf > 0.5:
                cvzone.putTextRect(img, f'{classNames[cls]}',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 7)

                # Play alert sound when detection happens
                if not pygame.mixer.music.get_busy():  # Prevent overlapping sounds
                    pygame.mixer.music.play()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
