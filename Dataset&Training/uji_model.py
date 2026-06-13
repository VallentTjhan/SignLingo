import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

actions = np.array(['halo', 'terimakasih', 'samasama', 'apakabar', 'senangbertemudenganmu', 'sampaijumpa', 'maaf'])
sequence_length = 60 

print("1. Membangunkan AI dari tidurnya...")

model = load_model('model_bisindo.h5')
print("AI Siap!")

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

sequence = [] 
threshold = 0.8 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

print("2. Menyalakan Kamera...")
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False                  
        results = holistic.process(image)                 
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:] 
        
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            if res[np.argmax(res)] > threshold: 
                kata_terdeteksi = actions[np.argmax(res)]
                
                cv2.rectangle(image, (0,0), (640, 60), (245, 117, 16), -1)
                cv2.putText(image, f"AI Menebak: {kata_terdeteksi.upper()}", (10,40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Uji Coba AI BISINDO', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

