import cv2
import numpy as np
import tensorflow as tf
import os
import sqlite3
import pyttsx3
import mediapipe as mp
from tensorflow.keras.models import load_model
from threading import Thread

# 1. Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Load Trained Keras Model (mediapipe-trained version, kept separate from the original)
model = load_model('Code/cnn_model_keras2_mp.keras')

# 3. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def get_image_size():
    img = cv2.imread('Code/gestures_mp/0/100.jpg', 0)
    if img is None:
        return (50, 50)
    return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize pixel values
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("Code/gesture_db_mp.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]
    return ""

def process_hand_crop(crop_img):
    # Convert crop to grayscale to match training data format
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Square padding to maintain aspect ratio
    h, w = gray.shape
    if w > h:
        pad = (w - h) // 2
        gray = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h > w:
        pad = (h - w) // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

    pred_probab, pred_class = keras_predict(model, gray)
    if pred_probab * 100 > 70:
        return get_pred_text_from_db(pred_class)
    return ""

def get_operator(pred_text):
    try:
        pred_text = int(pred_text)
    except:
        return ""
    operators = {1: "+", 2: "-", 3: "*", 4: "/", 5: "%", 6: "**", 7: ">>", 8: "<<", 9: "&", 0: "|"}
    return operators.get(pred_text, "")

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

def extract_hand_region(frame):
    """Detects hand using MediaPipe and returns crop bounding box."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    h, w, c = frame.shape
    crop = None
    bbox = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx < x_min: x_min = cx
                if cx > x_max: x_max = cx
                if cy < y_min: y_min = cy
                if cy > y_max: y_max = cy

            # Add margin around bounding box
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            if (x_max - x_min) > 20 and (y_max - y_min) > 20:
                crop = frame[y_min:y_max, x_min:x_max]
                bbox = (x_min, y_min, x_max, y_max)
            break

    return frame, crop, bbox

def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0

    while True:
        ret, frame = cam.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame, hand_crop, bbox = extract_hand_region(frame)
        
        old_text = text
        if hand_crop is not None:
            text = process_hand_crop(hand_crop)
            if old_text == text and text != "":
                count_same_frame += 1
            else:
                count_same_frame = 0

            if count_same_frame > 10:  # Faster detection threshold
                if len(text) == 1:
                    Thread(target=say_text, args=(text,)).start()
                word += text
                count_same_frame = 0

            # Draw green box around detected hand
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode (MediaPipe)", (120, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0))
        cv2.putText(blackboard, "Predicted text: " + str(text), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.8, (255, 255, 255))

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