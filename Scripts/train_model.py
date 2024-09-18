import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_anomaly_model():
    """Cria um modelo simples de CNN para detecção de anomalias."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_anomaly_model(model, train_dir, val_dir):
    """Treina o modelo usando os dados de treino e validação."""
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

    model.fit(train_generator, validation_data=val_generator, epochs=10)

    model.save('models/saved_models/anomaly_detector.h5')

# Exemplo de uso
train_dir = 'data/processed/train'
val_dir = 'data/processed/validation'
model = create_anomaly_model()
train_anomaly_model(model, train_dir, val_dir)
