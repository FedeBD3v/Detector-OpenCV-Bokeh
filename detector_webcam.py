import cv2
import time
import pandas as pd
from datetime import datetime

df = pd.DataFrame(columns=["Start", "End"])

face_cascade = cv2.CascadeClassifier('face_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('eyes_cascade/haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error al cargar los clasificadores.")
    exit()

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error al abrir la cámara.")
    exit()

status_list = [0, 0]
times = []

try:
    while True:
        status = 0
        ret, frame = video.read()
        
        if not ret:
            print("Error al capturar el frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            status = 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  

        status_list.append(status)

        if status_list[-1] == 1 and status_list[-2] == 0:
            times.append(datetime.now())
        if status_list[-1] == 0 and status_list[-2] == 1:
            times.append(datetime.now())

        cv2.imshow('Video - Reconocimiento Facial y de Ojos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if status == 1:
                times.append(datetime.now())
            break

finally:
    if len(times) % 2 != 0:
        times.append(datetime.now())
        times.append(datetime.now())  

    time_pairs = []
    for i in range(0, len(times), 2):
        time_pairs.append({"Start": times[i], "End": times[i + 1]})
    
    df = pd.DataFrame(time_pairs)
    df.to_csv("Times.csv", index=False)

    video.release()
    cv2.destroyAllWindows()
