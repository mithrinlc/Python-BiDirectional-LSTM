import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Model creation function
def create_model(sequence_length, num_features, num_output=4):
    model = Sequential([
        LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_output)  # Predicting next x, y position, speed, and angle
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load a pre-trained model
def load_model(file_path):
    return tf.keras.models.load_model(file_path)

# Save a trained model
def save_model(model, file_path):
    model.save(file_path)

# Normalize data
def normalize_data(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# Load and preprocess data
def load_and_preprocess_data(file_path, sequence_length=5):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    features = []
    for d in data:
        if d['event'] == 'move':
            features.append([
                float(d['data'].get('x', 0)), float(d['data'].get('y', 0)),
                float(d['data'].get('dx', 0)), float(d['data'].get('dy', 0)),
                float(d['data'].get('distance', 0)), float(d['data'].get('speed', 0)),
                float(d['data'].get('angle', 0)),
                float(d['data'].get('acceleration', 0)), float(d['data'].get('direction_change', 0)),
                float(d['data'].get('elapsed', 0)),
                0.0, 0.0  # Additional features for click data (not a click)
            ])
        elif d['event'] == 'click':
            click_type = 1.0 if d['data'].get('type') == 'down' else -1.0
            features.append([
                float(d['data'].get('x', 0)), float(d['data'].get('y', 0)), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, float(d['data'].get('interval', 0)),
                1.0, click_type  # Additional features for click data
            ])

    features = np.array(features, dtype=np.float32)
    features = normalize_data(features)

    x, y = [], []
    for i in range(len(features) - sequence_length):
        x.append(features[i:i + sequence_length])
        y.append(features[i + sequence_length][:4])  # Next x, y position, speed, and angle

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

# Train the model
def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return history

# Generate path to target
def generate_path_to_target(model, start, end, sequence_length=5, threshold=10):
    current_point = start
    movements = []
    while distance(current_point, end) > threshold:
        sequence = create_sequence(movements, sequence_length)
        next_movement = model.predict(sequence)
        movements.append(next_movement)
        current_point = update_position(current_point, next_movement)
    return movements

# Calculate distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Create a sequence input for the model from the movements
def create_sequence(movements, sequence_length):
    # Pad the sequence if necessary
    sequence = np.zeros((sequence_length, len(movements[0])))
    start_idx = max(0, len(movements) - sequence_length)
    for i, movement in enumerate(movements[start_idx:]):
        sequence[i] = movement
    return np.array([sequence], dtype=np.float32)

# Update the current point based on the movement
def update_position(current_point, movement):
    return current_point[0] + movement[0], current_point[1] + movement[1]
