import cv2
import numpy as np
import tensorflow as tf
import os
import sqlite3
import pyttsx3
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from threading import Thread

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model('Code/gesture_lstm_mp.keras')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

SEQ_LEN = 30
PRED_STRIDE = 5       # run a prediction every N frames once the buffer is full
CONF_THRESHOLD = 0.70


def extract_landmarks(frame):
    """Same normalization as create_gestures.py — wrist-relative, scale-normalized."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return frame, None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    h, w, c = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    bbox = (
        max(0, int(min(xs)) - 30), max(0, int(min(ys)) - 30),
        min(w, int(max(xs)) + 30), min(h, int(max(ys)) + 30),
    )

    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale

    return frame, pts.flatten(), bbox


def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("Code/gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]
    return ""


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

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame, landmarks, bbox = extract_landmarks(frame)

        if landmarks is not None:
            buffer.append(landmarks)
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        else:
            buffer.clear()

        frame_count += 1
        if len(buffer) == SEQ_LEN and frame_count % PRED_STRIDE == 0:
            seq = np.expand_dims(np.array(buffer), axis=0)  # shape (1, SEQ_LEN, 63)
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
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))

        res = np.hstack((frame, blackboard))
        cv2.imshow("Recognizing gesture", res)

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break
        elif keypress == ord('v'):
            is_voice_on = not is_voice_on

    return 0


def recognize():
    cam = cv2.VideoCapture(0)
    text_mode(cam)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()