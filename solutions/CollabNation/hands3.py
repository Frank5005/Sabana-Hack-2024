import cv2
import os
import numpy as np

# Cambiar valor
dataPath = r'C:/Users/frank/GitHub Projects/Sabana-Hack-2024/solutions/fotos'
peopleList =  os.listdir(dataPath)
print('Personas en la DB: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print("Thinking ...")

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)

        facesData.append(cv2.imread(personPath + '/' +  fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)

        cv2.imshow('image', image)
        cv2.waitKey(10)

    label = label + 1
cv2.destroyAllWindows()


print("Labels = ", labels)
print("Etiquetas: ", np.count_nonzero(np.array(labels)==0))
print("Etiquetas: ", np.count_nonzero(np.array(labels)==1))

# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# print("Training...")
# face_recognizer.train(facesData, np.array(labels))

# face_recognizer.write('ModeloFaceFrontalData.xml')
# print("Modelo Guardado")