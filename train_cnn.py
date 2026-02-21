import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 24
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "../dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "../dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="binary",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=8)

model.save("../model/eye_cnn.h5")
print(" Model saved")