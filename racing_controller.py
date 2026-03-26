import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


# ---------------- Key states ----------------
current_lr = None
nitro_on = False
drift_on = False


# ---------------- Key helpers ----------------
def hold_key(state, key, flag_name):
    global nitro_on, drift_on

    flag = nitro_on if flag_name == "nitro" else drift_on

    if state and not flag:
        pyautogui.keyDown(key)
        if flag_name == "nitro":
            nitro_on = True
        else:
            drift_on = True

    elif not state and flag:
        pyautogui.keyUp(key)
        if flag_name == "nitro":
            nitro_on = False
        else:
            drift_on = False


def press_lr(key):
    global current_lr
    if current_lr != key:
        if current_lr:
            pyautogui.keyUp(current_lr)
        pyautogui.keyDown(key)
        current_lr = key


def release_lr():
    global current_lr
    if current_lr:
        pyautogui.keyUp(current_lr)
        current_lr = None


# ---------------- Finger counter ----------------
def count_fingers(hand):
    tips = [8, 12, 16, 20]
    count = 0

    for tip in tips:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            count += 1

    return count


# ---------------- Main loop ----------------
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    steer_action = "STRAIGHT"
    nitro_active = False
    drift_active = False

    if results.multi_hand_landmarks:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape

            # ---- RED fingertip dots ----
            for tip in [4, 8, 12, 16, 20]:
                x = int(hand_landmarks.landmark[tip].x * w)
                y = int(hand_landmarks.landmark[tip].y * h)
                cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

            fingers = count_fingers(hand_landmarks)

            # -------- Left hand -> Steering --------
            if label == "Left":

                if fingers == 2:
                    press_lr('d')
                    steer_action = "RIGHT"

                elif fingers == 3:
                    press_lr('a')
                    steer_action = "LEFT"

                else:
                    release_lr()
                    steer_action = "STRAIGHT"

            # -------- Right hand -> Nitro / Drift --------
            if label == "Right":

                if fingers >= 4:
                    nitro_active = True

                elif fingers == 2:
                    drift_active = True

    else:
        release_lr()

    # apply keys
    hold_key(nitro_active, 'space', "nitro")
    hold_key(drift_active, 'shift', "drift")

    # -------- Optional preview --------
    cv2.putText(img, f"STEER: {steer_action}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if nitro_on:
        cv2.putText(img, "NITRO!", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if drift_on:
        cv2.putText(img, "DRIFT!", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    cv2.imshow("CV Racing Controller", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows() 