import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        frame_out = frame.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_iris = landmarks[468]
            right_iris = landmarks[473]

            left_iris_px = np.array([left_iris.x * w, left_iris.y * h])
            right_iris_px = np.array([right_iris.x * w, right_iris.y * h])

            eye_indices = [33, 133, 145, 153, 362, 263]
            xs = [landmarks[i].x * w for i in eye_indices]
            ys = [landmarks[i].y * h for i in eye_indices]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            lx_norm = (left_iris_px[0] - x_min) / (x_max - x_min + 1e-6)
            ly_norm = (left_iris_px[1] - y_min) / (y_max - y_min + 1e-6)
            rx_norm = (right_iris_px[0] - x_min) / (x_max - x_min + 1e-6)
            ry_norm = (right_iris_px[1] - y_min) / (y_max - y_min + 1e-6)

            avg_x = (lx_norm + rx_norm) / 2
            avg_y = (ly_norm + ry_norm) / 2

            direction = "Center"
            if avg_x < 0.38:
                direction = "Left"
            elif avg_x > 0.62:
                direction = "Right"
            elif avg_y < 0.38:
                direction = "Up"
            elif avg_y > 0.62:
                direction = "Down"

            cv2.circle(frame_out, tuple(left_iris_px.astype(int)), 3, (0,255,0), -1)
            cv2.circle(frame_out, tuple(right_iris_px.astype(int)), 3, (0,255,0), -1)
            cv2.putText(frame_out, f"Direction: {direction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.rectangle(frame_out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 1)

        cv2.imshow("Iris Movement Detection", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
