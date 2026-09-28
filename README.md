![Stars](https://img.shields.io/github/stars/VarunKashyapTV/Sign-Language-Interpreter-using-Deep-Learning)
![Forks](https://img.shields.io/github/forks/VarunKashyapTV/Sign-Language-Interpreter-using-Deep-Learning)
![Language](https://img.shields.io/github/languages/top/VarunKashyapTV/Sign-Language-Interpreter-using-Deep-Learning)

# Sign Language Interpreter using Deep Learning

A real-time sign language interpreter that reads a live webcam feed, recognizes signs (including signs that involve hand movement) and speaks them aloud. Hand tracking uses MediaPipe landmarks, and an LSTM classifies short landmark sequences.

Forked from and built on [harshbg/Sign-Language-Interpreter-using-Deep-Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning), then reworked from a histogram/contour + CNN pipeline into a MediaPipe + LSTM pipeline.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies and Tools](#technologies-and-tools)
* [Setup](#setup)
* [Process](#process)
* [Code Examples](#code-examples)
* [Features](#features)
* [Status](#status)
* [Future Work](#future-work)
* [Credits](#credits)
* [Contact](#contact)

## General info
Many people who are deaf or hard of hearing depend on interpreters for everyday communication. This project turns a webcam into a personal interpreter: it recognizes signs and outputs them as on-screen text and speech.

Many signs (for example "yes" and "no") depend on movement, which a single-image classifier cannot capture. This version classifies a 30-frame sequence of hand landmarks instead of one cropped image, so static and moving signs are handled by the same model.


## Technologies and Tools
* Python 3.11
* MediaPipe Hands (0.10.14)
* TensorFlow / Keras (2.16.1), LSTM
* OpenCV
* NumPy, scikit-learn
* pyttsx3 (text-to-speech)
* SQLite (gesture ID to name mapping)

## Setup
1. Install Python 3.11 (newer versions do not have TensorFlow wheels yet).
2. Create and activate a virtual environment, then install the pinned dependencies:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The versions are pinned on purpose: `numpy<2` avoids an ABI crash with OpenCV/TensorFlow, `mediapipe==0.10.14` still ships the `mp.solutions.hands` API, and `tensorflow==2.16.1` resolves to a protobuf version that MediaPipe also accepts.

## Process
Run steps 1 to 3 from inside the `Code` folder.

1. Run `create_gestures.py`. It lists the gestures already in the database and the next free ID, asks for a gesture ID and name, then captures 150 landmark sequences automatically while you perform the sign. Hold static signs steadily and repeat moving signs many times. Output: `gestures_mp/<id>.npy`. Press `Q` to stop early.
2. Run `load_images.py` to combine all captured sequences and split them into training, validation and test sets.
3. Run `cnn_model_train.py` to train the LSTM. The best model by validation accuracy is saved as `gesture_lstm_mp.keras`.
4. From the repository root, run `python Code/final.py`. The recognition window opens, uses your webcam to interpret the trained signs and speaks them. Press `V` to toggle voice and `Q` to quit.

Captured data, datasets and trained models are not stored in this repository, so you need to capture your own gestures before training.

## Code Examples
```python
# Model training using an LSTM on landmark sequences
SEQ_LEN = 30
FEATURES = 63  # 21 landmarks x (x, y, z)

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
    return model
```

## Features
* Real-time recognition from a standard webcam
* Handles both static signs and signs that involve movement
* Landmarks are normalized relative to the wrist and hand size, so the position and distance of your hand do not matter
* No lighting-sensitive calibration step
* Recognized signs are spoken aloud with text-to-speech
* Capture script shows the existing gestures and the next free ID every time it runs

## Status
Project is: in progress. Data capture and model training for the full gesture set are ongoing.

## Future Work
* Hardware integration for the PBL build (camera pan-tilt tracking, audio output)
* Larger vocabulary and more signers for robustness
* Feedback mechanism to correct wrong predictions
* Support for more sign languages

## Credits
* Original project: [harshbg/Sign-Language-Interpreter-using-Deep-Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning)
* Hand landmark detection: [MediaPipe](https://developers.google.com/mediapipe)

