import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Inicializar Mediapipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Ruta del dataset
dataset_path = 'C:/Users/frank/.cache/kagglehub/datasets/realtimear/hand-wash-dataset/versions/4/HandWashDataset/HandWashDataset'
steps = os.listdir(dataset_path)

# Listas para almacenar secuencias y etiquetas
sequences = []
labels = []

# Recorrer cada carpeta de pasos
for step in steps:
    step_path = os.path.join(dataset_path, step)
    for video_file in os.listdir(step_path):
        cap = cv2.VideoCapture(os.path.join(step_path, video_file))
        frame_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir el fotograma a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                # Extraer landmarks y almacenarlos en la secuencia
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                frame_sequence.append(landmarks)

                # Limita la secuencia a una longitud fija (por ejemplo, 30 fotogramas)
                if len(frame_sequence) == 30:
                    sequences.append(np.array(frame_sequence).flatten())  # Aplanar la secuencia para sklearn
                    labels.append(step)  # Usa el nombre de la carpeta como etiqueta
                    frame_sequence = []

        cap.release()

# Convertir a numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Codificar etiquetas
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Diccionarios para almacenar las métricas de cada modelo
f1_scores = {}
roc_auc_scores = {}

# Realizar el entrenamiento en One-vs-All para cada paso
for step in np.unique(encoded_labels):
    print(f"\nEntrenando modelo para detectar el paso: {label_encoder.inverse_transform([step])[0]}")

    # Crear etiquetas binarias (1 si es el paso actual, 0 si no)
    binary_labels = (encoded_labels == step).astype(int)

    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(sequences, binary_labels, test_size=0.2, random_state=42)

    # Entrenar modelo RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilidades para ROC AUC

    # Calcular métricas
    rf_f1 = f1_score(y_test, rf_predictions)
    try:
        rf_roc_auc = roc_auc_score(y_test, rf_proba)
    except ValueError:
        rf_roc_auc = np.nan  # Si solo hay una clase en y_test, manejar con NaN

    # Almacenar métricas
    f1_scores[label_encoder.inverse_transform([step])[0]] = rf_f1
    roc_auc_scores[label_encoder.inverse_transform([step])[0]] = rf_roc_auc

    print(f"F1 Score para {label_encoder.inverse_transform([step])[0]}: {rf_f1}")
    print(f"ROC AUC para {label_encoder.inverse_transform([step])[0]}: {rf_roc_auc}")

# Guardar cada modelo entrenado para cada paso
for step, model in zip(np.unique(encoded_labels), [rf_model]):
    filename = f"model_{label_encoder.inverse_transform([step])[0]}.joblib"
    joblib.dump(model, filename)
    print(f"Modelo para {label_encoder.inverse_transform([step])[0]} guardado como {filename}")
