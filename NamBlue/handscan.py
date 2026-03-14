import cv2 as cv
import mediapipe as mp
import math # Thêm thư viện toán học để tính khoảng cách

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv.VideoCapture(0)
        self.is_pinched = False

    def get_signal(self):
        ret, frame = self.cap.read()
        if not ret: return False
        
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb)
        
        trigger = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lấy tọa độ đầu ngón cái (ID 4) và đầu ngón trỏ (ID 8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Tính khoảng cách giữa 2 đầu ngón tay
                distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                # Ngưỡng (threshold): 0.05 là khoảng cách khi 2 ngón gần như chạm nhau
                if distance < 0.05: 
                    if not self.is_pinched:
                        trigger = True
                        self.is_pinched = True
                else:
                    self.is_pinched = False
                
                # Vẽ để theo dõi
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        cv.imshow('Camera Control', frame)
        cv.waitKey(1)
        return trigger

    def cleanup(self):
        self.cap.release()
        cv.destroyAllWindows()