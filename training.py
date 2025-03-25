import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Setze die Pfade zu deinem Dataset
train_dir = 'C:/Users/apoll/OneDrive/Desktop/Programming/KI2425/RoboFlow/dataset_emotions/train'
valid_dir = 'C:/Users/apoll/OneDrive/Desktop/Programming/KI2425/RoboFlow/dataset_emotions/valid'
test_dir = 'C:/Users/apoll/OneDrive/Desktop/Programming/KI2425/RoboFlow/dataset_emotions/test'

# Bildgrößen und Batch-Größen
img_size = (128, 128)  # Größe der Bilder, die ins Modell eingehen
batch_size = 32

# Datenvorbereitung mit ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Einfaches CNN-Modell ohne vortrainierte Modelle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')  # 7 Klassen, also 7 Ausgabeneuronen
])

# Kompiliere das Modell
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Trainiere das Modell
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=40,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size
)

# Speichere das Modell als .h5-Datei
model.save('emotions_model.h5')

# Evaluieren des Modells auf den Testdaten
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc:.4f}')
