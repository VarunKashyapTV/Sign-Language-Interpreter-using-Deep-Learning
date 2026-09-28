import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEQ_LEN = 30
FEATURES = 63  # 21 landmarks x 3 (x, y, z)


def lstm_model(num_of_classes):
    model = Sequential()
    model.add(Input(shape=(SEQ_LEN, FEATURES)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    filepath = "gesture_lstm_mp.keras"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    return model, [checkpoint]


def train():
    with open("train_sequences_mp", "rb") as f:
        train_sequences = np.array(pickle.load(f), dtype=np.float32)
    with open("train_labels_mp", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_sequences_mp", "rb") as f:
        val_sequences = np.array(pickle.load(f), dtype=np.float32)
    with open("val_labels_mp", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    train_labels_cat = to_categorical(train_labels)
    val_labels_cat = to_categorical(val_labels)

    num_of_classes = train_labels_cat.shape[1]
    print(f"Dataset target classes count: {num_of_classes}")

    model, callbacks_list = lstm_model(num_of_classes)
    model.summary()

    model.fit(
        train_sequences,
        train_labels_cat,
        validation_data=(val_sequences, val_labels_cat),
        epochs=30,
        batch_size=32,
        callbacks=callbacks_list,
    )

    scores = model.evaluate(val_sequences, val_labels_cat, verbose=0)
    print("LSTM Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    train()
    K.clear_session()