import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "iris_dataset.csv")
FRAMES_PER_LABEL = 30
DELAY_BETWEEN_FRAMES = 0.03

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(OUT_FILE):
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "l_x", "l_y", "r_x", "r_y", "label"])

print("""
-----------------------------------------------
 IRIS DATA COLLECTION (AUTO-RECORD MODE)

 Instructions:
  - Look in one direction (left, right, up, down, center, blink)
  - Press the corresponding key and hold briefly:
      l = left
      r = right
      u = up
      d = down
      c = center
      b = blink
  - Press 'q' to quit
-----------------------------------------------
""")

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

            l_norm = ((left_iris_px[0] - x_min) / (x_max - x_min + 1e-6),
                      (left_iris_px[1] - y_min) / (y_max - y_min + 1e-6))
            r_norm = ((right_iris_px[0] - x_min) / (x_max - x_min + 1e-6),
                      (right_iris_px[1] - y_min) / (y_max - y_min + 1e-6))

            cv2.circle(frame_out, tuple(left_iris_px.astype(int)), 3, (0,255,0), -1)
            cv2.circle(frame_out, tuple(right_iris_px.astype(int)), 3, (0,255,0), -1)
            cv2.rectangle(frame_out, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0,0), 1)
            cv2.putText(frame_out, "Press key for label (l/r/u/d/c/b)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Iris Data Collection", frame_out)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            print("\nData collection ended.")
            break

        # Label keys
        key_map = {
            ord('l'): "left",
            ord('r'): "right",
            ord('u'): "up",
            ord('d'): "down",
            ord('c'): "center",
            ord('b'): "blink"
        }

        # If pressed key is a label AND face detected, auto-record multiple frames
        if key in key_map and results.multi_face_landmarks:
            label = key_map[key]
            print(f"[+] Recording {FRAMES_PER_LABEL} frames for '{label}'...")
            for _ in range(FRAMES_PER_LABEL):
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_inner = face_mesh.process(rgb)
                if results_inner.multi_face_landmarks:
                    lm = results_inner.multi_face_landmarks[0].landmark
                    l_iris = lm[468]; r_iris = lm[473]
                    l_px = np.array([l_iris.x * w, l_iris.y * h])
                    r_px = np.array([r_iris.x * w, r_iris.y * h])

                    xs = [lm[i].x * w for i in eye_indices]
                    ys = [lm[i].y * h for i in eye_indices]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    l_norm = ((l_px[0] - x_min) / (x_max - x_min + 1e-6),
                              (l_px[1] - y_min) / (y_max - y_min + 1e-6))
                    r_norm = ((r_px[0] - x_min) / (x_max - x_min + 1e-6),
                              (r_px[1] - y_min) / (y_max - y_min + 1e-6))

                    timestamp = time.time()
                    with open(OUT_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, l_norm[0], l_norm[1], r_norm[0], r_norm[1], label])
                time.sleep(DELAY_BETWEEN_FRAMES)
            print(f"[+] Done recording '{label}'!")








