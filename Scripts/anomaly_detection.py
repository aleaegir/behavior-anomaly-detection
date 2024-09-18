import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_anomaly_model(model_path):
    """Carrega o modelo de detecção de anomalias."""
    return load_model(model_path)

def detect_anomalies(model, frame):
    """Detecta anomalias em um frame de vídeo."""
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    predictions = model.predict(frame)
    return predictions[0][0] > 0.5  # Exemplo: se a previsão for maior que 0.5, é uma anomalia

# Exemplo de uso
model_path = 'models/saved_models/anomaly_detector.h5'
model = load_anomaly_model(model_path)

frame = cv2.imread('data/processed/frames/frame_0.jpg')
is_anomaly = detect_anomalies(model, frame)

if is_anomaly:
    print('Anomalia detectada!')
else:
    print('Nenhuma anomalia detectada.')
