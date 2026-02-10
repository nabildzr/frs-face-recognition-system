import cv2
import face_recognition
import os
import pickle
import numpy as np

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
ENCODING_THRESHOLD = 0.5  # Stricter than default 0.6
GEOMETRY_THRESHOLD = 0.5 # Adjusted for weighted MSE (needs tuning)

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
                if isinstance(data, dict) and "encoding" in data and "landmarks" in data:
                    known_faces[name] = data
                    print(f"Loaded {name} (Advanced format)")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"Skipping {name}: Old format (landmarks only). Please re-encode using face_encoder.py.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return known_faces

# --- Preprocessing ---
def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# --- Geometry & Alignment ---
def get_eye_aspect_ratio(eye_landmarks):
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (v1 + v2) / (2.0 * h)

def align_face(landmarks):
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    all_points = []
    parts = ["chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]
    for part in parts:
        all_points.extend(landmarks[part])
    all_points = np.array(all_points, dtype=np.float32)
    
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
        pts1 = align_face(landmarks1)
        pts2 = align_face(landmarks2)
        
        norm1 = normalize_points(pts1)
        norm2 = normalize_points(pts2)
        
        if norm1.shape != norm2.shape:
            return float('inf')
        
        weights = np.ones(len(norm1))
        
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


# --- Verification API (Server-side) ---
# Only handles face encoding matching + geometry verification.
# Liveness detection is handled by the client (online_frame_tracker.py).
def verify_face_sequence(frame_data_list, known_faces=None):
    """
    Verifies a sequence of face frame data against known faces.
    Only performs encoding matching and geometry verification.
    
    Args:
        frame_data_list: List of dicts, each containing:
            - 'encoding': np.array (128-d face encoding)
            - 'landmarks': dict (face_recognition landmarks format)
        known_faces: Dict of known faces (from load_known_faces). 
                     If None, will load from default directory.
    
    Returns:
        dict with keys:
            - 'verified': bool (geometry match passed)
            - 'name': str (matched name or "Unknown")
            - 'status': str (human-readable status)
            - 'avg_geometry_score': float
            - 'frames_analyzed': int
    """
    if known_faces is None:
        known_faces = load_known_faces()
    
    if not known_faces:
        return {
            'verified': False,
            'name': 'Unknown',
            'status': 'NO KNOWN FACES LOADED',
            'avg_geometry_score': float('inf'),
            'frames_analyzed': 0
        }
    
    if not frame_data_list or len(frame_data_list) < 5:
        return {
            'verified': False,
            'name': 'Unknown',
            'status': f'INSUFFICIENT FRAMES ({len(frame_data_list) if frame_data_list else 0}/5)',
            'avg_geometry_score': float('inf'),
            'frames_analyzed': len(frame_data_list) if frame_data_list else 0
        }
    
    # Step 1: Identify the person from encodings (majority vote)
    name_votes = {}
    for frame_data in frame_data_list:
        encoding = frame_data['encoding']
        
        matches = face_recognition.compare_faces(
            [data['encoding'] for data in known_faces.values()],
            encoding,
            tolerance=ENCODING_THRESHOLD
        )
        
        if True in matches:
            face_distances = face_recognition.face_distance(
                [data['encoding'] for data in known_faces.values()],
                encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = list(known_faces.keys())[best_match_index]
                name_votes[name] = name_votes.get(name, 0) + 1
    
    if not name_votes:
        return {
            'verified': False,
            'name': 'Unknown',
            'status': 'NO ENCODING MATCH',
            'avg_geometry_score': float('inf'),
            'frames_analyzed': len(frame_data_list)
        }
    
    # Best match by majority vote
    best_match_name = max(name_votes, key=name_votes.get)
    known_landmarks = known_faces[best_match_name]['landmarks']
    
    # Step 2: Geometry verification (average score across all frames)
    geo_scores = []
    for frame_data in frame_data_list:
        landmarks = frame_data['landmarks']
        geo_score = weighted_geometry_distance(landmarks, known_landmarks)
        geo_scores.append(geo_score)
    
    avg_geo = sum(geo_scores) / len(geo_scores)
    geometry_pass = avg_geo < GEOMETRY_THRESHOLD
    
    if geometry_pass:
        status = "GEOMETRY MATCH"
    else:
        status = "FACE MISMATCH"
    
    return {
        'verified': geometry_pass,
        'name': best_match_name,
        'status': status,
        'avg_geometry_score': avg_geo,
        'frames_analyzed': len(frame_data_list)
    }
