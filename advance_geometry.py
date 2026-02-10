import cv2
import face_recognition
import os
import glob
import pickle
import numpy as np
from collections import deque

# --- Configuration ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
KNOWN_FACES_DIR = "known_faces"
ENCODING_THRESHOLD = 0.5  # Stricter than default 0.6
GEOMETRY_THRESHOLD = 0.5 # Adjusted for weighted MSE (needs tuning)
HISTORY_LEN = 5
BLINK_THRESHOLD = 0.2  # Eye Aspect Ratio threshold
CONSECUTIVE_FRAMES = 3

# --- Data Loading ---
def load_known_faces(directory=KNOWN_FACES_DIR):
    known_faces = {}
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return known_faces

    for filename in os.listdir(directory):
        if filename.endswith(".bin"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                
                name = os.path.splitext(filename)[0]
                # Handle both old format (list of landmarks) and new format (dict)
                if isinstance(data, dict) and "encoding" in data and "landmarks" in data:
                    known_faces[name] = data
                    print(f"Loaded {name} (Advanced format)")
                elif isinstance(data, list) and len(data) > 0:
                    # Fallback or skip
                    print(f"Skipping {name}: Old format (landmarks only). Please re-encode using face_encoder.py.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return known_faces

# --- Preprocessing ---
def preprocess_frame(frame):
    # Histogram Equalization on Y channel of LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# --- Geometry & Alignment ---
def get_eye_aspect_ratio(eye_landmarks):
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # p1..p6 are 0..5 in 0-indexed list
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (v1 + v2) / (2.0 * h)

def align_face(landmarks):
    # Simple alignment based on eyes
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Center of rotation is the midpoint between eyes
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Flatten landmarks to list of points
    all_points = []
    parts = ["chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]
    for part in parts:
        all_points.extend(landmarks[part])
    all_points = np.array(all_points, dtype=np.float32)
    
    # Apply rotation
    ones = np.ones(shape=(len(all_points), 1))
    points_ones = np.hstack([all_points, ones])
    rotated_points = M.dot(points_ones.T).T
    
    return rotated_points

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if rms > 0:
        return centered / rms
    return centered

def weighted_geometry_distance(landmarks1, landmarks2):
    try:
        # Align both
        pts1 = align_face(landmarks1)
        pts2 = align_face(landmarks2)
        
        norm1 = normalize_points(pts1)
        norm2 = normalize_points(pts2)
        
        if norm1.shape != norm2.shape:
            return float('inf')
            
        # Weights (Simple hardcoded indices based on standard 72-point Dlib model)
        # Chin (0-16): Low
        # Eyebrows (17-26): Medium
        # Nose (27-35): High
        # Eyes (36-47): High
        # Mouth (48-67): Medium/Low
        
        weights = np.ones(len(norm1))
        
        # Indices depend on the order in align_face
        # Chin: 17 points
        # Left Eyebrow: 5 points
        # Right Eyebrow: 5 points
        # Nose Bridge: 4 points
        # Nose Tip: 5 points
        # Left Eye: 6 points
        # Right Eye: 6 points
        # Top Lip: 12 points
        # Bottom Lip: 12 points
        
        idx = 0
        weights[idx:idx+17] = 0.5; idx += 17 # Chin
        weights[idx:idx+5] = 0.8; idx += 5   # L Brow
        weights[idx:idx+5] = 0.8; idx += 5   # R Brow
        weights[idx:idx+4] = 1.0; idx += 4   # Nose Bridge
        weights[idx:idx+5] = 1.0; idx += 5   # Nose Tip
        weights[idx:idx+6] = 1.2; idx += 6   # L Eye
        weights[idx:idx+6] = 1.2; idx += 6   # R Eye
        weights[idx:idx+12] = 0.6; idx += 12 # Top Lip
        weights[idx:idx+12] = 0.6; idx += 12 # Bottom Lip
        
        diff = norm1 - norm2
        weighted_mse = np.mean(np.sum(diff**2, axis=1) * weights)
        
        return weighted_mse
    except Exception as e:
        print(f"Error in weighted_geometry_distance: {e}")
        return float('inf')

# --- Tracking Class ---
class FaceTracker:
    def __init__(self, name):
        self.name = name
        self.scores = deque(maxlen=HISTORY_LEN)
        self.blink_counter = 0
        self.is_alive = False

    def update(self, score, ear):
        self.scores.append(score)
        
        if ear < BLINK_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= 1: # Simple blink detected
                self.is_alive = True
            self.blink_counter = 0

    def get_average_score(self):
        if not self.scores:
            return float('inf')
        return sum(self.scores) / len(self.scores)
    
    def get_status(self):
        avg = self.get_average_score()
        status = "MATCH" if avg < GEOMETRY_THRESHOLD else "NO MATCH"
        liveness = "LIVE" if self.is_alive else "SPOOF?"
        return f"{self.name} | Score: {avg:.4f} | {status} | {liveness}"

# --- Main ---
def main():
    print("Loading known faces...")
    known_faces = load_known_faces()
    if not known_faces:
        print("No advanced format known faces found. Please run face_encoder.py first.")
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    active_tracker = FaceTracker("Unknown")
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 15
    
    print("Starting video loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Preprocess
        processed_frame = preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
                
                for (top, right, bottom, left), encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                    
                    # 1. Encoding Filter
                    matches = face_recognition.compare_faces(
                        [data['encoding'] for data in known_faces.values()], 
                        encoding, 
                        tolerance=ENCODING_THRESHOLD
                    )
                    
                    best_match_name = "Unknown"
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(
                            [data['encoding'] for data in known_faces.values()], 
                            encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            best_match_name = list(known_faces.keys())[best_match_index]
                    
                    if best_match_name != "Unknown":
                        # 2. Geometry Verification
                        known_landmarks = known_faces[best_match_name]['landmarks']
                        geo_score = weighted_geometry_distance(landmarks, known_landmarks)
                        
                        # 3. Liveness (EAR)
                        left_ear = get_eye_aspect_ratio(landmarks['left_eye'])
                        right_ear = get_eye_aspect_ratio(landmarks['right_eye'])
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        # Update Tracker
                        if active_tracker.name != best_match_name:
                            active_tracker = FaceTracker(best_match_name)
                            
                        active_tracker.update(geo_score, avg_ear)
                        
                        status_text = active_tracker.get_status()
                        color = (0, 255, 0) if "MATCH" in status_text and "LIVE" in status_text else (0, 165, 255)
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw some landmarks for visual feedback
                        for point in landmarks["left_eye"] + landmarks["right_eye"] + landmarks["nose_tip"]:
                            cv2.circle(frame, point, 2, (0, 255, 255), -1)

                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # No faces
                pass

        cv2.imshow("Advanced Face Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
