import cv2
import numpy as np
import os, sqlite3, random
import mediapipe as mp

image_x, image_y = 50, 50

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils


def init_create_folder_database():
    # create the folder and database if not exist
    if not os.path.exists("gestures_mp"):
        os.mkdir("gestures_mp")
    if not os.path.exists("gesture_db_mp.db"):
        conn = sqlite3.connect("gesture_db_mp.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db_mp.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, '%s')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == "y":
            cmd = "UPDATE gesture SET g_name = '%s' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("Doing nothing...")
            return
    conn.commit()


def extract_hand_region(frame):
    """Same logic as final.py's extract_hand_region — keeps training/inference
    crops consistent. Returns (frame_with_drawing, color_crop_or_None, bbox_or_None)."""
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


def process_hand_crop(crop_img):
    """Mirrors final.py's process_hand_crop, but returns the processed image
    instead of a prediction, since here we're saving it, not classifying it."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w > h:
        pad = (w - h) // 2
        gray = cv2.copyMakeBorder(gray, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h > w:
        pad = (h - w) // 2
        gray = cv2.copyMakeBorder(gray, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

    return cv2.resize(gray, (image_x, image_y))


def store_images(g_id):
    total_pics = 1200
    cam = cv2.VideoCapture(0)

    create_folder("gestures_mp/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, img = cam.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))  # must match final.py's frame size

        img, hand_crop, bbox = extract_hand_region(img)

        processed = None
        if hand_crop is not None:
            processed = process_hand_crop(hand_crop)
            if bbox:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if processed is not None and flag_start_capturing and frames > 50:
            pic_no += 1
            rand = random.randint(0, 10)
            out_img = cv2.flip(processed, 1) if rand % 2 == 0 else processed
            cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
            cv2.imwrite("gestures_mp/" + str(g_id) + "/" + str(pic_no) + ".jpg", out_img)
            cv2.imshow("Cropped (saved) hand", out_img)

        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", img)

        keypress = cv2.waitKey(1)
        if keypress == ord("c"):
            if not flag_start_capturing:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing:
            frames += 1
        if pic_no == total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()


init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)