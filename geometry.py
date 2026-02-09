import cv2
import face_recognition
import os
import glob
import pickle
os.environ["QT_QPA_PLATFORM"] = "xcb"
import numpy as np

# Buka kamera
def load_face_geometry(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data[0] # Return the first face's landmarks
        return None
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def normalize_landmarks(landmarks):
    points = []
    parts = ["chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]
    for part in parts:
        if part in landmarks:
            points.extend(landmarks[part])
    
    points = np.array(points, dtype=np.float32)
    
    if len(points) == 0:
        return None

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    rms_distance = np.sqrt(np.mean(np.sum(centered_points**2, axis=1)))
    if rms_distance > 0:
        normalized_points = centered_points / rms_distance
    else:
        normalized_points = centered_points

    return normalized_points

def compare_geometries(geom1, geom2):
    norm1 = normalize_landmarks(geom1)
    norm2 = normalize_landmarks(geom2)

    if norm1 is None or norm2 is None:
        return float('inf')

    if norm1.shape != norm2.shape:
        return float('inf')

    mse = np.mean(np.sum((norm1 - norm2)**2, axis=1))
    return mse

known_geometry = load_face_geometry("known_faces/putra.bin")
if known_geometry is not None:
    print("Loaded known_faces/putra.bin")

cap = cv2.VideoCapture(0)

frame_count = 0
PROCESS_EVERY_N_FRAMES = 20
face_landmarks_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Ubah ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah
        face_locations = face_recognition.face_locations(rgb_frame)

        # Ambil landmark geometri wajah
        face_landmarks_list = face_recognition.face_landmarks(
            rgb_frame,
            face_locations
        )

        if known_geometry is not None:
            for face_landmarks in face_landmarks_list:
                score = compare_geometries(known_geometry, face_landmarks)
                print(f"Comparison Score: {score:.4f} - {'MATCH' if score < 0.05 else 'NO MATCH'}")

    for landmarks in face_landmarks_list:
        # Gambar titik mata
        for point in landmarks["left_eye"]:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        for point in landmarks["right_eye"]:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        # Gambar hidung
        for point in landmarks["nose_tip"]:
            cv2.circle(frame, point, 2, (255, 0, 0), -1)

        # Gambar mulut
        for point in landmarks["top_lip"]:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)

        for point in landmarks["bottom_lip"]:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)

        # Gambar rahang (bentuk wajah)
        for point in landmarks["chin"]:
            cv2.circle(frame, point, 3, (255, 255, 0), -1)

        # Gambar alis
        for point in landmarks["left_eyebrow"]:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)
        for point in landmarks["right_eyebrow"]:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

        # Gambar tulang hidung
        for point in landmarks["nose_bridge"]:
            cv2.circle(frame, point, 2, (255, 0, 255), -1)

    cv2.imshow("Face Geometry Detection", frame)
    
    # Simpan otomatis jika ada wajah
    if face_landmarks_list:
        with open("face_geometry.bin", "wb") as f:
            pickle.dump(face_landmarks_list, f)
        print(f"Data geometri wajah tersimpan otomatis ({len(face_landmarks_list)} wajah)")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
