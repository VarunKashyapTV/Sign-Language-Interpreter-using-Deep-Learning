import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle

import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

MODEL_FILE = "gesture_lstm_mp.keras"


def lstm_model(num_of_classes, seq_len, features):
    model = Sequential()
    model.add(Input(shape=(seq_len, features)))
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
    checkpoint = ModelCheckpoint(
        MODEL_FILE, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
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

    seq_len, features = train_sequences.shape[1], train_sequences.shape[2]
    print(f"Sequences: {train_sequences.shape[0]} train, {val_sequences.shape[0]} validation; "
          f"each {seq_len} frames x {features} features")

    # same class count for both splits, even if one split happens to miss the highest ID
    num_of_classes = int(max(train_labels.max(), val_labels.max())) + 1
    print(f"Dataset target classes count: {num_of_classes}")
    train_labels = to_categorical(train_labels, num_of_classes)
    val_labels = to_categorical(val_labels, num_of_classes)

    model, callbacks_list = lstm_model(num_of_classes, seq_len, features)
    model.summary()

    model.fit(
        train_sequences,
        train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=30,
        batch_size=32,
        callbacks=callbacks_list,
    )

    scores = model.evaluate(val_sequences, val_labels, verbose=0)
    print("LSTM Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    train()
    K.clear_session()