from flask import Flask, render_template, request, redirect, url_for
import threading
import cv2
import os
import face_recognition as fr
import pandas as pd

app = Flask(__name__)

# Ruta de la carpeta de imágenes de los alumnos
base_dir = os.path.dirname(os.path.realpath(__file__))
imageFacesPath = base_dir + "/fotos_alumnos"

facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(os.path.join(imageFacesPath, file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f_coding = fr.face_encodings(image, num_jitters=10)[0]
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split(".")[0])

registro_df = pd.DataFrame(columns=['Nombre', 'Archivo_JPG'])

# Control de detección
detener_deteccion = threading.Event()


def deteccion_rostros():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while not detener_deteccion.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        orig = frame.copy()
        faces = faceClassif.detectMultiScale(frame, 1.1, 5)

        # (Resto de tu código)...
        
    cap.release()
    cv2.destroyAllWindows()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'start' in request.form:
            detener_deteccion.clear()
            threading.Thread(target=deteccion_rostros).start()
        elif 'stop' in request.form:
            detener_deteccion.set()
        return redirect(url_for('resultados'))
    return render_template('index.html')


@app.route('/resultados')
def resultados():
    return render_template('result.html', datos=registro_df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
