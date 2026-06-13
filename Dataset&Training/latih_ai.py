import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('Dataset_BISINDO') 
actions = np.array(['halo', 'terimakasih', 'samasama', 'apakabar', 'senangbertemudenganmu', 'sampaijumpa', 'maaf']) 
no_sequences = 60
sequence_length = 60

print("1. Sedang menyusun buku flipbook dari folder...")

sequences, labels = [], []

label_map = {label:num for num, label in enumerate(actions)}

for action in actions:
    for sequence in range(no_sequences):
        window = [] 
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window) 
        labels.append(label_map[action]) 

X = np.array(sequences)

y = to_categorical(labels).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("Data siap! Bentuk data AI kita:", X.shape)

print("2. Membangun Otak LSTM...")

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 258), unroll=True))

model.add(LSTM(128, return_sequences=True, activation='relu', unroll=True))

model.add(LSTM(64, return_sequences=False, activation='relu', unroll=True))

model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("3. MULAI PROSES BELAJAR (Training)...")

model.fit(X_train, y_train, epochs=85)

print("4. BELAJAR SELESAI! Menyimpan otak AI ke file...")

model.save('model_bisindo.h5')
print("Model berhasil disimpan dengan nama 'model_bisindo.h5'!")

