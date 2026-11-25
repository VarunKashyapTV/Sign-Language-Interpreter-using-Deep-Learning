import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sqlite3
from collections import deque
from threading import Thread

import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

from hand_features import SEQ_LEN, FEATURES, build_features, extract_features

BASE = os.path.dirname(os.path.abspath(__file__))     # works from any folder
PRED_STRIDE = 5          # predict every N frames once the buffer is full
CONF_THRESHOLD = 0.70    # accept a prediction only above this confidence
HAND_GONE_FRAMES = 15    # after this many frames without a hand, the same word may be spoken again

engine = pyttsx3.init()
engine.setProperty("rate", 150)

model = load_model(os.path.join(BASE, "gesture_lstm_mp.keras"))
if tuple(model.input_shape[1:]) != (SEQ_LEN, FEATURES):
    raise SystemExit(
        f"The saved model expects input {tuple(model.input_shape[1:])} but the code now produces "
        f"({SEQ_LEN}, {FEATURES}). Recapture the gestures and retrain the model."
    )


def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect(os.path.join(BASE, "gesture_db.db"))
    row = conn.execute("SELECT g_name FROM gesture WHERE g_id=?", (int(pred_class),)).fetchone()
    conn.close()
    return row[0] if row else ""


is_voice_on = True


def say_text(text):
    if not is_voice_on:
        return
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass

    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()


def text_mode(cam):
    global is_voice_on
    buffer = deque(maxlen=SEQ_LEN)
    text = ""
    word = ""
    frame_count = 0
    last_spoken = ""
    missing = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame, found = extract_features(frame)

        vec = build_features(found)
        if vec is not None:
            buffer.append(vec)
            missing = 0
        else:
            buffer.clear()
            missing += 1
            if missing > HAND_GONE_FRAMES:
                last_spoken = ""
                text = ""

        frame_count += 1
        if len(buffer) == SEQ_LEN and frame_count % PRED_STRIDE == 0:
            seq = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)   # (1, SEQ_LEN, FEATURES)
            pred_probab = model.predict(seq, verbose=0)[0]
            pred_class = int(np.argmax(pred_probab))
            confidence = float(np.max(pred_probab))

            if confidence > CONF_THRESHOLD:
                predicted = get_pred_text_from_db(pred_class)
                if predicted and predicted != last_spoken:
                    Thread(target=say_text, args=(predicted,)).start()
                    word += predicted + " "
                    last_spoken = predicted
                text = predicted
            else:
                text = ""

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode (MediaPipe + LSTM)", (60, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
        cv2.putText(blackboard, "Predicted: " + str(text), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, f"Hands in view: {len(found)}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))

        cv2.imshow("Recognizing gesture", np.hstack((frame, blackboard)))

        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            break
        elif keypress == ord("v"):
            is_voice_on = not is_voice_on

    return 0


def recognize():
    cam = cv2.VideoCapture(0)
    text_mode(cam)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()