import cv2
import os
import face_recognition as fr
import pandas as pd
import threading

from flask import (Flask, 
                   redirect, 
                   render_template, 
                   request, 
                   session, 
                   url_for,)


app = Flask(__name__)
app.secret_key = 'atendi'

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_frame = frame[y:y+h, x:x+w]
            roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            encoding_current_frame = fr.face_encodings(roi_frame)

            for encoding in encoding_current_frame:
                matches = fr.compare_faces(facesEncodings, encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = facesNames[match_index]
                    if not any(registro_df['Nombre'] == name):
                        registro_df.loc[len(registro_df)] = [name, f"{name.replace(' ', '_')}.jpg"]
    cap.release()
    cv2.destroyAllWindows()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'start' in request.form:
            detener_deteccion.clear()
            threading.Thread(target=deteccion_rostros).start()
            session['estado'] = 'Reconociendo rostros'
        elif 'stop' in request.form:
            detener_deteccion.set()
            session['estado'] = 'Alumnos registrados'
        return redirect(url_for('resultados'))
    return render_template('index.html')


@app.route('/resultados')
def resultados():
    estado = session.get('estado', 'Error')
    return render_template('result.html', datos=registro_df.to_dict(orient='records'), estado=estado)


@app.route('/finalizar', methods=['POST'])
def finalizar():
    detener_deteccion.set()
    if registro_df.empty:
        session['estado'] = 'No se reconocieron rostros'
    else:
        session['estado'] = 'Se reconocieron rostros'
    return redirect(url_for('resultados'))


@app.route('/reiniciar', methods=['POST'])
def reiniciar():
    global registro_df
    # Limpia el DataFrame
    registro_df = pd.DataFrame(columns=['Nombre', 'Archivo_JPG'])
    # Inicia de nuevo la detección de rostros
    detener_deteccion.clear()
    threading.Thread(target=deteccion_rostros).start()
    session['estado'] = 'Reconociendo rostros'
    return redirect(url_for('resultados'))


@app.route('/confirmar', methods=['POST'])
def confirmar():
    # Detiene la detección de rostros
    detener_deteccion.set()
    session['estado'] = 'Presentes confirmados'
    return redirect(url_for('resultados'))


if __name__ == '__main__':
    app.run(debug=True)