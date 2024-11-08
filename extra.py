import cv2

# Cargar el clasificador de rostros y el clasificador de ojos
face_cascade = cv2.CascadeClassifier('face_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('eyes_cascade/haarcascade_eye.xml')

# Capturar el video de la cámara
video = cv2.VideoCapture(0)

while True:
    # Leer cada cuadro del video
    ret, frame = video.read()
    
    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el cuadro en escala de grises
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Región de interés (ROI) para detectar ojos dentro del rostro detectado
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar ojos en la región de interés
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            # Calcular el centro del ojo
            center = (ex + ew // 2, ey + eh // 2)
            # Calcular el radio como la mitad del ancho (o altura) del área del ojo detectado
            radius = ew // 2
            cv2.circle(roi_color, center, radius, (0, 255, 0), 2)  # Verde para ojos


    # Mostrar el cuadro con los rostros y ojos detectados
    cv2.imshow('Video - Reconocimiento Facial y de Ojos', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el recurso de la cámara y cerrar las ventanas
video.release()
cv2.destroyAllWindows()
