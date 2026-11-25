import os
import sqlite3
from collections import deque

import cv2
import numpy as np

from hand_features import SEQ_LEN, FEATURES, build_features, extract_features

TOTAL_SEQUENCES = 150    # sequences to capture per gesture
STRIDE = 5               # save a new sequence every STRIDE frames (sliding window)
DB_FILE = "gesture_db.db"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def init_create_folder_database():
    if not os.path.exists("gestures_mp"):
        os.mkdir("gestures_mp")
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gesture "
        "( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
    )
    conn.commit()
    conn.close()


def list_existing_gestures():
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Change its name? (y/n): ")
        if choice.strip().lower() == "y":
            conn.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
        else:
            print("Keeping the existing name.")
    conn.commit()
    conn.close()


def draw_instructions(img, g_id, g_name, seq_no, total, n_hands, need):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"Gesture #{g_id}: {g_name}", (10, 24), FONT, 0.65, (255, 255, 255), 2)
    msg = f"Perform the sign with {'ONE hand' if need == 1 else 'BOTH hands'} - capture is automatic"
    cv2.putText(img, msg, (10, 50), FONT, 0.55, (0, 255, 0), 2)
    cv2.putText(img, "Q to quit early", (10, 72), FONT, 0.5, (200, 200, 200), 1)
    cv2.putText(img, f"Saved sequences: {seq_no}/{total}", (10, 98), FONT, 0.6, (255, 255, 0), 2)
    status = f"Hands in view: {n_hands} (need {need})"
    color = (0, 255, 0) if n_hands == need else (0, 0, 255)
    (tw, _), _ = cv2.getTextSize(status, FONT, 0.6, 2)
    cv2.putText(img, status, (w - tw - 10, 24), FONT, 0.6, color, 2)
    return img


def capture_sequences(g_id, g_name, need):
    cam = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LEN)
    sequences = []
    frame_count = 0
    failures = 0

    print("\n--- Capture controls ---")
    print(f"  Perform the sign with {'ONE hand' if need == 1 else 'BOTH hands'} repeatedly in front of the camera.")
    print(f"  A sequence is saved every {STRIDE} frames once {SEQ_LEN} frames are buffered.")
    print("  Frames with the wrong number of hands are ignored. Q quits early (saved sequences are kept).")
    print(f"Will stop automatically after {TOTAL_SEQUENCES} sequences.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            failures += 1
            if failures > 100:
                print("Camera is not giving frames. Close other apps using it and try again.")
                break
            continue
        failures = 0
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        frame, found = extract_features(frame)
        if len(found) == need:
            buffer.append(build_features(found))
        else:
            buffer.clear()          # wrong number of hands breaks the current sequence

        frame_count += 1
        if len(buffer) == SEQ_LEN and frame_count % STRIDE == 0:
            sequences.append(np.array(buffer, dtype=np.float32))     # shape (SEQ_LEN, FEATURES)

        frame = draw_instructions(frame, g_id, g_name, len(sequences), TOTAL_SEQUENCES, len(found), need)
        cv2.imshow("Capturing gesture", frame)

        if cv2.waitKey(1) == ord("q"):
            print(f"Stopped early. Saved {len(sequences)}/{TOTAL_SEQUENCES} sequences.")
            break
        if len(sequences) >= TOTAL_SEQUENCES:
            print(f"Done - saved {TOTAL_SEQUENCES} sequences for gesture #{g_id} ({g_name}).")
            break

    cam.release()
    cv2.destroyAllWindows()

    if sequences:
        data = np.array(sequences, dtype=np.float32)
        out_path = os.path.join("gestures_mp", f"{g_id}.npy")
        np.save(out_path, data)
        print(f"Saved to {out_path} - shape {data.shape} (sequences, frames, {FEATURES} features)")


if __name__ == "__main__":
    init_create_folder_database()
    list_existing_gestures()
    g_id = int(input("Enter gesture no.: "))
    g_name = input("Enter gesture name/text: ").strip()
    need = 0
    while need not in (1, 2):
        raw = input("How many hands does this sign use? (1 or 2): ").strip()
        need = int(raw) if raw in ("1", "2") else 0
    out = os.path.join("gestures_mp", f"{g_id}.npy")
    if os.path.exists(out):
        if input(f"{out} already exists. Overwrite? (y/n): ").strip().lower() != "y":
            raise SystemExit("Cancelled.")
    store_in_db(g_id, g_name)
    capture_sequences(g_id, g_name, need)