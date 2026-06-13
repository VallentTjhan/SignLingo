import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH = os.path.join('Dataset_BISINDO') 

actions = np.array(['halo', 'terimakasih', 'samasama', 'apakabar', 'senangbertemudenganmu', 'sampaijumpa', 'maaf']) 

no_sequences = 60 
sequence_length = 60 

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass 

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Menunggu kamera memunculkan gambar...")
                    cv2.waitKey(500)
                    continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False                  
                results = holistic.process(image)                 
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                if frame_num == 0: 
                    cv2.putText(image, 'BERSIAP...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Merekam kata "{action}" | Take ke-{sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Kamera BISINDO', image)
                    
                    cv2.waitKey(3000) 
                else: 
                    cv2.putText(image, f'Merekam kata "{action}" | Take ke-{sequence}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Kamera BISINDO', image)
                
                keypoints = extract_keypoints(results) 
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num)) 
                np.save(npy_path, keypoints) 

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
cap.release()
cv2.destroyAllWindows()
