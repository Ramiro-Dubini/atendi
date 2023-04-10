import cv2
import os
import face_recognition as fr
import pandas as pd

# Ruta de la carpeta de imágenes de los alumnos
imageFacesPath = "C:/Users/Ramiro/Documents/Code/script/fotos_alumnos"

# Codificar los rostros extraídos
facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(os.path.join(imageFacesPath, file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f_coding = fr.face_encodings(image, num_jitters=10)[0]  # Aumentamos el número de jitters para mejorar la codificación
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split(".")[0])

# LEYENDO VIDEO
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.1, 5)

    for (x, y, w, h) in faces:
        face = orig[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_encodings = fr.face_encodings(face, num_jitters=10)
        if len(face_encodings) > 0:
            actual_face_encoding = face_encodings[0]  # Tomamos la primera codificación
            result = fr.compare_faces(facesEncodings, actual_face_encoding)
            if True in result:
                index = result.index(True)
                name = facesNames[index]
                color = (125, 220, 0)
            else:
                name = "Desconocido"
                color = (50, 50, 255)
        else:
            name = "Desconocido"
            color = (50, 50, 255)

        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Detecta la tecla "esc" (código ASCII 27)
        break
    
cap.release()
cv2.destroyAllWindows()