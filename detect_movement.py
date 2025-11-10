import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
#
model = tf.keras.models.load_model('models/iris_model.h5')

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
##
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Starting real-time iris movement detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        l_x = face_landmarks.landmark[468].x
        l_y = face_landmarks.landmark[468].y
        r_x = face_landmarks.landmark[473].x
        r_y = face_landmarks.landmark[473].y

        # Normalize and prepare input
        X = np.array([[l_x, l_y, r_x, r_y]])

        prediction = model.predict(X)
        predicted_class = np.argmax(prediction)
        movement_label = label_encoder.inverse_transform([predicted_class])[0]

        h, w, _ = frame.shape
        for idx in [468, 473]:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f'Movement: {movement_label}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Iris Movement Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











