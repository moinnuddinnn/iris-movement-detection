import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('models/iris_model.h5')

# Load the label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start capturing from the camera
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
            # Extract iris landmarks (MediaPipe indexes 468-473)
            iris_coords = []
            for idx in range(468, 473):
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                iris_coords.append([x, y])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if len(iris_coords) == 5:
                iris_array = np.array(iris_coords).flatten()
                iris_array = iris_array / np.array([w, h] * (len(iris_coords)))  # Normalize
                iris_array = np.expand_dims(iris_array, axis=0)

                # Predict movement
                prediction = model.predict(iris_array)
                predicted_class = np.argmax(prediction)
                movement_label = label_encoder.inverse_transform([predicted_class])[0]

                # Display prediction
                cv2.putText(frame, f'Movement: {movement_label}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Iris Movement Detection', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
