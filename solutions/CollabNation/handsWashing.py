"""Código del programa de detección del lavado de manos correcto"""

# pylint: disable=C0103
# pylint: disable=C0305
# pylint: disable=W0311
# pylint: disable=C0116
# pylint: disable=C0303
# pylint: disable=C0321
# pylint: disable=C0301
# pylint: disable=W0621

import cv2
import mediapipe as mp

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterate through the landmarks to find the bounding box coordinates
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
# Para la detección de objetos

def get_hand_bounding_box(landmarks, frame_shape, margin=50):
    x_coords = [lm.x * frame_shape[1] for lm in landmarks.landmark]
    y_coords = [lm.y * frame_shape[0] for lm in landmarks.landmark]
    min_x, max_x = max(0, int(min(x_coords) - margin)), min(frame_shape[1], int(max(x_coords) + margin))
    min_y, max_y = max(0, int(min(y_coords) - margin)), min(frame_shape[0], int(max(y_coords) + margin))
    return (min_x, min_y), (max_x, max_y)

# Detecta cambios en el área de la mano

def detect_object_in_proximity(frame, bbox, threshold=1000):
    (min_x, min_y), (max_x, max_y) = bbox
    hand_area = frame[min_y:max_y, min_x:max_x]
    gray_hand_area = cv2.cvtColor(hand_area, cv2.COLOR_BGR2GRAY)
    _, thresh_hand_area = cv2.threshold(gray_hand_area, 80, 255, cv2.THRESH_BINARY)
    count_white_pixels = cv2.countNonZero(thresh_hand_area)
    return count_white_pixels > threshold

# Fin jejej



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#mp_objects = mp.solutions

cap = cv2.VideoCapture(0)

cap.set(3,1920)
cap.set(4,1080)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks):
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Draw bounding box
                draw_bounding_box(image, hand_landmarks)

                index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
                
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                int(hand_landmarks.landmark[4].y * image_height))
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                                int(hand_landmarks.landmark[2].y * image_height))
                
                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                int(hand_landmarks.landmark[12].y * image_height))
                
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                int(hand_landmarks.landmark[10].y * image_height))
                
                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                int(hand_landmarks.landmark[16].y * image_height))
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                int(hand_landmarks.landmark[14].y * image_height))
                
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                int(hand_landmarks.landmark[20].y * image_height))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                int(hand_landmarks.landmark[18].y * image_height))
                
                wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))
                
                ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                int(hand_landmarks.landmark[5].y * image_height))
                
                
                # Detección de objetos
                
                bbox = get_hand_bounding_box(hand_landmarks, image.shape)
                object_detected = detect_object_in_proximity(image, bbox)

                # Dibuja el bounding box alrededor de la mano
                cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)

                if object_detected:
                    cv2.putText(image, "Objeto detectado cerca de la mano, alejarlo si es posible", (bbox[0][0], bbox[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                
                # Detección a través de las distancias
                
                if abs(thumb_tip[1] - index_finger_pip[1]) <45 \
                    and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and abs(thumb_tip[1] - ring_finger_pip[1]) < 30\
                    and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                    cv2.putText(image, 'A', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                   
                elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
                        middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<40:
                    cv2.putText(image, 'B', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
                    index_finger_tip[1] - middle_finger_pip[1]<0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
                        index_finger_tip[1] - index_finger_pip[1] > 0:
                   cv2.putText(image, 'C', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                
                elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
                    and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
                    and  pinky_pip[1] - pinky_tip[1]<0\
                    and index_finger_pip[1] - index_finger_tip[1]>0:
                    cv2.putText(image, 'D', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                   
                elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
                        and abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                            thumb_tip[1] - index_finger_tip[1] > 0 \
                            and thumb_tip[1] - middle_finger_tip[1] > 0 \
                            and thumb_tip[1] - ring_finger_tip[1] > 0 \
                            and thumb_tip[1] - pinky_tip[1] > 0:

                    cv2.putText(image, 'E', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                elif  pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                    ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
                        and abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) <65:

                    cv2.putText(image, 'F', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                print("pulgar", thumb_tip[1])
                print("dedo indice",index_finger_tip[1])
                
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()

