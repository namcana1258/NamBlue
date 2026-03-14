import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = 0
            lm = handLms.landmark

            tips = [ 8, 12, 16, 20]

            for tip in tips:
                if lm[tip].y < lm[tip - 2].y:
                    fingers += 1

            # ngón cái
            if lm[4].x > lm[3].x:
                fingers += 1

            cv.putText(frame, f"Fingers: {fingers}", (20,50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv.imshow("hand", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()