import cv2
from matplotlib import pyplot
import imutils
from mtcnn.mtcnn import MTCNN
import os

direccion = 'C:/Users/frank/GitHub Projects/Sabana-Hack-2024/solutions/CollabNation/fotos'
nombre = "persona_sin_tapabocas"
# nombre = "persona_con_tapabocas"
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)


detector = MTCNN()
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()

    caras = detector.detect_faces(copia)

    for i in range(len(caras)):
        x1,y1,ancho,alto = caras[i]['box']
        x2,y2 = x1 + ancho, y1 + alto
        cara_reg = frame[y1:y2, x1:x2]
        cara_reg = cv2.resize(cara_reg, (150, 200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(carpeta + "/rostro_{}.jpg".format(count), cara_reg)
        count = count  + 1

    cv2.imshow("Training", frame)

    t = cv2.waitKey(1)
    if t == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
