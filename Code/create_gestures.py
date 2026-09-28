import cv2
import numpy as np
import os, sqlite3
import mediapipe as mp
from collections import deque

SEQ_LEN = 30            # frames per sequence (~1 sec at 30fps)
NUM_LANDMARKS = 21
FEATURES = NUM_LANDMARKS * 3   # x, y, z per landmark = 63
TOTAL_SEQUENCES = 150   # sequences to capture per gesture
STRIDE = 5               # save a new sequence every STRIDE frames (sliding window)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils


def init_create_folder_database():
    if not os.path.exists("gestures_mp"):
        os.mkdir("gestures_mp")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()


def list_existing_gestures():
    conn = sqlite3.connect("gesture_db.db")
    rows = conn.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id").fetchall()
    conn.close()
    if rows:
        print("\nExisting gestures in DB:")
        for g_id, g_name in rows:
            print(f"  {g_id}: {g_name}")
        next_id = rows[-1][0] + 1
    else:
        print("\nNo gestures in DB yet.")
        next_id = 0
    print(f"Next available ID: {next_id}\n")
    return next_id


def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
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


def extract_landmarks(frame):
    """Runs MediaPipe on a frame, draws the skeleton, and returns a normalized
    63-length landmark vector (or None if no hand detected). Normalization:
    subtract the wrist (landmark 0) so it's translation-invariant, then divide
    by the largest landmark distance from the wrist so it's scale-invariant."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return frame, None

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    wrist = pts[0].copy()
    pts -= wrist

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale

    return frame, pts.flatten()  # shape (63,)


def draw_instructions(img, g_id, g_name, seq_no, total_seqs, hand_detected):
    h, w = img.shape[:2]
    panel_h = 100
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"Gesture #{g_id}: {g_name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(img, "Perform the sign repeatedly - capture is automatic", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, "Q to quit early", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Saved sequences: {seq_no}/{total_seqs}", (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    hand_text = "Hand detected" if hand_detected else "No hand detected"
    (tw, _), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, hand_text, (w - tw - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    return img


def capture_sequences(g_id, g_name):
    cam = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LEN)
    sequences = []
    frame_count = 0

    print("\n--- Capture controls ---")
    print("  Perform the gesture repeatedly in front of the camera.")
    print(f"  Capturing happens automatically every {STRIDE} frames once the buffer fills.")
    print("  Q : quit early (saved sequences are kept)")
    print(f"Will stop automatically after {TOTAL_SEQUENCES} sequences are saved.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        frame, landmarks = extract_landmarks(frame)
        hand_detected = landmarks is not None

        if hand_detected:
            buffer.append(landmarks)
        else:
            buffer.clear()  # a dropped hand breaks the current sequence

        frame_count += 1
        if len(buffer) == SEQ_LEN and frame_count % STRIDE == 0:
            sequences.append(np.array(buffer))  # shape (SEQ_LEN, 63)

        frame = draw_instructions(
            frame, g_id, g_name, len(sequences), TOTAL_SEQUENCES, hand_detected
        )
        cv2.imshow("Capturing gesture", frame)

        keypress = cv2.waitKey(1)
        if keypress == ord("q"):
            print(f"Stopped early. Saved {len(sequences)}/{TOTAL_SEQUENCES} sequences.")
            break
        if len(sequences) >= TOTAL_SEQUENCES:
            print(f"Done — saved {TOTAL_SEQUENCES} sequences for gesture #{g_id} ({g_name}).")
            break

    cam.release()
    cv2.destroyAllWindows()

    if sequences:
        out_path = os.path.join("gestures_mp", f"{g_id}.npy")
        np.save(out_path, np.array(sequences))
        print(f"Saved to {out_path} — shape {np.array(sequences).shape}")


if __name__ == "__main__":
    init_create_folder_database()
    list_existing_gestures()
    g_id = input("Enter gesture no.: ")
    g_name = input("Enter gesture name/text: ")
    store_in_db(g_id, g_name)
    print(
        "\nBoth static and moving signs are captured the same way here — "
        "just perform the sign (hold it if static, repeat it if it involves motion) "
        "in front of the camera; sequences are captured automatically.\n"
    )
    capture_sequences(g_id, g_name)