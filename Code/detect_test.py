"""Check hand detection for different people and lighting.
Runs MediaPipe on the raw camera image AND on a contrast-boosted copy and counts how often each finds a hand.
Hold the hand(s) in view for about 15 seconds, then press q."""
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_detector():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)


det_raw, det_enh = make_detector(), make_detector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def count(result):
    return len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0


cam = cv2.VideoCapture(0)
frames = raw_hits = enh_hits = 0
while True:
    ret, frame = cam.read()
    if not ret:
        continue
    frame = cv2.resize(cv2.flip(frame, 1), (640, 480))
    raw = det_raw.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enh = det_enh.process(cv2.cvtColor(enhance(frame), cv2.COLOR_BGR2RGB))
    n_raw, n_enh = count(raw), count(enh)
    frames += 1
    raw_hits += n_raw > 0
    enh_hits += n_enh > 0
    if enh.multi_hand_landmarks:
        for hl in enh.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    brightness = int(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    cv2.putText(frame, f"raw: {n_raw} hand(s)   boosted: {n_enh} hand(s)   brightness: {brightness}/255",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "q = finish", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Detection test", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
if frames:
    print(f"Frames: {frames}")
    print(f"Hand found - raw image: {100 * raw_hits / frames:.0f}%   contrast-boosted: {100 * enh_hits / frames:.0f}%")