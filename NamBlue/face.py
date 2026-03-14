import cv2 as cv
import mediapipe as mp

# Khởi tạo mediapipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

mp_draw = mp.solutions.drawing_utils

# Mở webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
            )

    cv.imshow("Face Landmark", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()