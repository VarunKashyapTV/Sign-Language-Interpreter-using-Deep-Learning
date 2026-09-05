import os
import pickle
import cv2
import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Suppress TensorFlow logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_image_size():
    img_path = os.path.join("gestures", "1", "100.jpg")
    img = cv2.imread(img_path, 0)
    if img is None:
        return (50, 50)
    return img.shape


image_x, image_y = get_image_size()


def cnn_model(num_of_classes):
    model = Sequential()

    model.add(Input(shape=(image_x, image_y, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_of_classes, activation="softmax"))

    # Adam optimizer works vastly better than raw SGD for image tasks
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Native .keras format recommended for current TensorFlow/Keras versions
    filepath = "cnn_model_keras2.keras"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )

    return model, [checkpoint]


def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f), dtype=np.float32)
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f), dtype=np.float32)
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # 1. Normalize pixels from [0, 255] to [0.0, 1.0]
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    train_images = np.reshape(
        train_images, (train_images.shape[0], image_x, image_y, 1)
    )
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    num_of_classes = val_labels.shape[1]
    print(f"Dataset target classes count: {num_of_classes}")

    model, callbacks_list = cnn_model(num_of_classes)
    model.summary()

    # 2. Reduced batch_size to 64 for faster, smoother updates
    model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=15,
        batch_size=64,
        callbacks=callbacks_list,
    )

    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    train()
    K.clear_session()
