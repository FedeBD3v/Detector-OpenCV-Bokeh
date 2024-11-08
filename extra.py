import cv2


face_cascade = cv2.CascadeClassifier('face_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('eyes_cascade/haarcascade_eye.xml')

video = cv2.VideoCapture(0)

while True:
   
    ret, frame = video.read()
    
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
      
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

     
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:

            center = (ex + ew // 2, ey + eh // 2)
     
            radius = ew // 2
            cv2.circle(roi_color, center, radius, (0, 255, 0), 2)  


  
    cv2.imshow('Video - Reconocimiento Facial y de Ojos', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
