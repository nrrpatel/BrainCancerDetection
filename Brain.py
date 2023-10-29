import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.src.backend_config import image_data_format
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

ROOT_DIR = "/Users/nikunjpatel/Desktop/Data"
number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    if not dir.startswith('.'):  # Check if the directory is not a hidden file
        directory_path = os.path.join(ROOT_DIR, dir)
        if os.path.isdir(directory_path):
            number_of_images[dir] = len(os.listdir(directory_path))

number_of_images.items()

def dataFolder(p, split):
    if not os.path.exists("./" + p):
        os.mkdir('./' + p)
        for dir in os.listdir(ROOT_DIR):
            if not dir.startswith('.'):  # Skip hidden files/directories
                os.makedirs("./" + p + "/" + dir)
                for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)), size=(math.floor(split * number_of_images[dir]) - 5), replace=False):
                    O = os.path.join(ROOT_DIR, dir, img)
                    D = os.path.join("./" + p, dir, img)  # Correct the destination path
                    shutil.copy(O, D)
                    os.remove(O)
    else:
        print(f"{p} Folder exists")


dataFolder("train", 0.7)
dataFolder("test", 0.15)
dataFolder("val", 0.15)
# CNN MODEL

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))

model.add(Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

# Preparing our data using Data Generator
def preprocessingImages(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, rescale=1 / 255, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

    return image

#
# path = "/Users/nikunjpatel/PycharmProjects/BrainTumorDetection/archive/Training"
# train_data = preprocessingImages(path)
#
# def preprocessingImages2(path):
#     image_data = ImageDataGenerator(rescale=1 / 255)
#     image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')
#
#     return image
#
# path = "/Users/nikunjpatel/PycharmProjects/BrainTumorDetection/archive/Testing"
# test_data = preprocessingImages2(path)
#
# path = "/Users/nikunjpatel/PycharmProjects/BrainTumorDetection/archive/Validation"
# val_data = preprocessingImages2(path)
#
# hs = model.fit(train_data, steps_per_epoch=8, epochs=30, verbose=1, validation_data=val_data, validation_steps=16)