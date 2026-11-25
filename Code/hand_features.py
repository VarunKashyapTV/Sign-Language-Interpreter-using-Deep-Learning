"""Shared hand-feature code used by create_gestures.py, final.py and detect_test.py.
Keeping it in ONE place guarantees that capture and live recognition preprocess identically."""
import cv2
import numpy as np
import mediapipe as mp

SEQ_LEN = 30                       # frames per sequence (about 1 second)
HAND_FEATURES = 63                 # 21 landmarks x (x, y, z)
FEATURES = 2 * HAND_FEATURES + 2   # slot 0 + slot 1 + (dx, dy) between the two wrists = 128
DET_CONF = 0.5                     # detection / tracking confidence (lower than before: finds dim or dark hands)
ENHANCE = True                     # contrast boost before detection (same setting for capture and live)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=DET_CONF,
    min_tracking_confidence=DET_CONF,
)
mp_draw = mp.solutions.drawing_utils
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame):
    """Local contrast boost (CLAHE on the lightness channel). Helps dark hands and dim rooms."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def hand_shape(pts):
    """Wrist-relative, size-normalised landmarks of one hand -> (63 numbers, raw wrist, hand size)."""
    pts = np.asarray(pts, dtype=np.float32)
    wrist = pts[0].copy()
    rel = pts - wrist
    scale = float(np.max(np.linalg.norm(rel, axis=1)))
    if scale > 1e-6:
        rel = rel / scale
    return rel.flatten(), wrist, scale


def build_features(found):
    """found = list of (label, pts) with pts of shape (21, 3), at most 2 hands.
    Layout of the returned 128-number vector:
      [0:63]    hand in slot 0 (the hand on the LEFT of the picture when two hands are shown)
      [63:126]  hand in slot 1 (the hand on the RIGHT of the picture)
      [126:128] offset between the two wrists (dx, dy) in hand sizes - this is what lets the
                model tell apart two-hand signs such as one hand on top of the other.
    One hand only: it goes in slot 0 if MediaPipe says Left, else slot 1; the rest stays zero."""
    if not found:
        return None
    found = found[:2]
    vec = np.zeros(FEATURES, np.float32)
    shapes = [hand_shape(p) for _, p in found]
    if len(found) == 2:
        order = sorted(range(2), key=lambda i: shapes[i][1][0])       # leftmost wrist first
        for slot, i in enumerate(order):
            vec[slot * HAND_FEATURES:(slot + 1) * HAND_FEATURES] = shapes[i][0]
        w0, w1 = shapes[order[0]][1], shapes[order[1]][1]
        size = max(shapes[0][2], shapes[1][2], 1e-6)
        vec[2 * HAND_FEATURES:] = np.clip((w1[:2] - w0[:2]) / size / 3.0, -2.0, 2.0)
    else:
        slot = 0 if found[0][0] == "Left" else 1
        vec[slot * HAND_FEATURES:(slot + 1) * HAND_FEATURES] = shapes[0][0]
    return vec


def extract_features(frame):
    """Runs MediaPipe (on a contrast-boosted copy), draws the skeletons on `frame`,
    and returns (frame, found) where found = [(label, pts), ...] for every hand seen."""
    src = enhance(frame) if ENHANCE else frame
    results = hands.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    found = []
    if results.multi_hand_landmarks:
        for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
            found.append((hd.classification[0].label, pts))   # frame is already mirrored, so labels are correct
    return frame, found