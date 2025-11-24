import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
FEATURES_DIR = 'data/processed_features/'
LSTM_MODEL_PATH = 'models/lstm/theft_detector_lstm.h5'
SEQUENCE_LENGTH = 30  # Number of frames to look at in one sequence

# --- Load Data ---
def load_data():
    X, y = [], []
    for label, category in enumerate(['normal', 'theft']): # normal=0, theft=1
        category_path = os.path.join(FEATURES_DIR, category)
        for filename in os.listdir(category_path):
            if filename.endswith('.npy'):
                features = np.load(os.path.join(category_path, filename))
                # Create sequences of length SEQUENCE_LENGTH
                for i in range(0, len(features) - SEQUENCE_LENGTH, 10): # Stride of 10
                    sequence = features[i:i + SEQUENCE_LENGTH]
                    X.append(sequence)
                    y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
print(f"Loaded {len(X)} sequences.")

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Build LSTM Model ---
# Get the shape from the training data
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential([
    Masking(mask_value=0., input_shape=input_shape),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train Model ---
print("Training LSTM model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# --- Save Model ---
model.save(LSTM_MODEL_PATH)
print(f"Model saved to {LSTM_MODEL_PATH}")