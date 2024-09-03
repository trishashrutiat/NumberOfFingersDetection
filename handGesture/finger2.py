import cv2
import time
import os
import HandTrackingModule as htm
import serial
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize serial communication
try:
    arduino = serial.Serial(port='COM15', baudrate=9600, timeout=1)
    logging.info("Connected to Arduino")
except Exception as e:
    logging.error(f"Failed to connect to Arduino: {e}")
    arduino = None

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "finger"
myList = os.listdir(folderPath)
logging.info(f"Image list: {myList}")

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (200, 200))  # Resize to 200x200
    overlayList.append(image)

logging.info(f"Loaded {len(overlayList)} overlay images.")

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        logging.warning("Failed to capture image")
        continue

    try:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            logging.info(f"Detected fingers: {totalFingers}")

            # Send the number of fingers detected to Arduino
            if arduino:
                try:
                    arduino.write(str(totalFingers).encode())
                    time.sleep(0.05)  # Add a small delay
                except Exception as e:
                    logging.error(f"Failed to send data to Arduino: {e}")

            if 0 <= totalFingers - 1 < len(overlayList):
                img[0:200, 0:200] = overlayList[totalFingers - 1]

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
