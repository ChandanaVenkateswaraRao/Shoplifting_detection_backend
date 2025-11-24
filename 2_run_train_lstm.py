# backend/2_run_train_lstm.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# --- Configuration ---
FEATURES_DIR = 'data/processed_features/'
# Save the new model with a different name to avoid overwriting the old one immediately
NEW_LSTM_MODEL_PATH = 'models/lstm/theft_detector_lstm_v2.h5' 
SEQUENCE_LENGTH = 30
INPUT_SHAPE = (SEQUENCE_LENGTH, 60) # Should be (SEQUENCE_LENGTH, MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE)

# --- Load Data ---
def load_data():
    X, y = [], []
    for label, category in enumerate(['normal', 'theft']):
        category_path = os.path.join(FEATURES_DIR, category)
        for filename in tqdm(os.listdir(category_path), desc=f"Loading '{category}' features"):
            if filename.endswith('.npy'):
                features = np.load(os.path.join(category_path, filename))
                if features.shape[1] != INPUT_SHAPE[1]:
                    print(f"\nSkipping {filename} due to incorrect feature size. Expected {INPUT_SHAPE[1]}, got {features.shape[1]}")
                    continue
                for i in range(0, len(features) - SEQUENCE_LENGTH, 10):
                    sequence = features[i:i + SEQUENCE_LENGTH]
                    X.append(sequence)
                    y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
print(f"\nLoaded {len(X)} sequences for training.")

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} sequences, validating on {len(X_test)} sequences.")

# --- Build LSTM Model ---
model = Sequential([
    Masking(mask_value=0., input_shape=INPUT_SHAPE),
    LSTM(128, return_sequences=True), # Increased complexity for more data
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train Model ---
print("\n--- Starting LSTM model training ---")
# Consider increasing epochs for more data, e.g., 20-30
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# --- Save Model ---
model.save(NEW_LSTM_MODEL_PATH)
print(f"\n--- Training complete! New model saved to {NEW_LSTM_MODEL_PATH} ---")