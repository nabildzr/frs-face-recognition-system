import cv2
import face_recognition
import os
import pickle
import numpy as np
import base64
import json
from flask import Flask, request, jsonify

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
ENCODING_THRESHOLD = 0.5
GEOMETRY_THRESHOLD = 0.5

app = Flask(__name__)

# Global known faces (loaded once)
known_faces = {}

# --- Data Loading ---
def load_known_faces(directory=KNOWN_FACES_DIR):
    faces = {}
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return faces

    for filename in os.listdir(directory):
        if filename.endswith(".bin"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                
                name = os.path.splitext(filename)[0]
                if isinstance(data, dict) and "encoding" in data and "landmarks" in data:
                    faces[name] = data
                    print(f"Loaded {name} (Advanced format)")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"Skipping {name}: Old format.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return faces

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
        weights[idx:idx+17] = 0.5; idx += 17
        weights[idx:idx+5] = 0.8; idx += 5
        weights[idx:idx+5] = 0.8; idx += 5
        weights[idx:idx+4] = 1.0; idx += 4
        weights[idx:idx+5] = 1.0; idx += 5
        weights[idx:idx+6] = 1.2; idx += 6
        weights[idx:idx+6] = 1.2; idx += 6
        weights[idx:idx+12] = 0.6; idx += 12
        weights[idx:idx+12] = 0.6; idx += 12
        
        diff = norm1 - norm2
        weighted_mse = np.mean(np.sum(diff**2, axis=1) * weights)
        
        return weighted_mse
    except Exception as e:
        print(f"Error in weighted_geometry_distance: {e}")
        return float('inf')


# --- Deserialization Helpers ---
def decode_frame_data(serialized_list):
    """Decode base64-encoded frame data back to numpy arrays and landmark dicts."""
    frame_data_list = []
    for item in serialized_list:
        encoding = np.frombuffer(base64.b64decode(item['encoding']), dtype=np.float64)
        
        # Convert landmark values from lists back to tuples
        landmarks = {}
        for key, points in item['landmarks'].items():
            landmarks[key] = [tuple(p) for p in points]
        
        frame_data_list.append({
            'encoding': encoding,
            'landmarks': landmarks,
        })
    return frame_data_list


# --- HTTP API Endpoint ---
@app.route('/verify', methods=['POST'])
def verify():
    """
    POST /verify
    Body (JSON):
        {
            "frames": [
                {
                    "encoding": "<base64 encoded numpy array>",
                    "landmarks": { "left_eye": [[x,y], ...], ... }
                },
                ...
            ]
        }
    
    Returns (JSON):
        {
            "verified": bool,
            "name": str,
            "status": str,
            "avg_geometry_score": float,
            "frames_analyzed": int
        }
    """
    try:
        data = request.get_json(force=True)
        
        print(f"\n[API Server] === Incoming /verify request ===")
        print(f"[API Server] Content-Type: {request.content_type}")
        print(f"[API Server] Content-Length: {request.content_length}")
        
        if not data or 'frames' not in data:
            print("[API Server] ERROR: Missing 'frames' key in request body")
            return jsonify({
                'verified': False,
                'name': 'Unknown',
                'status': 'INVALID REQUEST - missing frames',
                'avg_geometry_score': 9999.0,
                'frames_analyzed': 0
            }), 400
        
        serialized_frames = data['frames']
        print(f"[API Server] Received {len(serialized_frames)} frame(s)")
        
        # Debug first frame structure
        if serialized_frames:
            first = serialized_frames[0]
            print(f"[API Server] First frame keys: {list(first.keys())}")
            print(f"[API Server] Encoding length (base64): {len(first.get('encoding', ''))}")
            print(f"[API Server] Landmark keys: {list(first.get('landmarks', {}).keys())}")
        
        # Decode
        print("[API Server] Decoding frame data...")
        frame_data_list = decode_frame_data(serialized_frames)
        print(f"[API Server] Decoded {len(frame_data_list)} frame(s) successfully")
        
        # Debug decoded data
        if frame_data_list:
            fd0 = frame_data_list[0]
            print(f"[API Server] First decoded encoding shape: {fd0['encoding'].shape}, dtype: {fd0['encoding'].dtype}")
            print(f"[API Server] First decoded landmark keys: {list(fd0['landmarks'].keys())}")
        
        # Verify
        print("[API Server] Running verify_face_sequence...")
        result = verify_face_sequence(frame_data_list, known_faces)
        print(f"[API Server] Result: {result}")
        print(f"[API Server] === Request complete ===\n")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"\n[API Server] !!! ERROR in /verify !!!")
        print(f"[API Server] Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"[API Server] !!! END ERROR !!!\n")
        return jsonify({
            'verified': False,
            'name': 'Unknown',
            'status': f'SERVER ERROR: {str(e)}',
            'avg_geometry_score': 9999.0,
            'frames_analyzed': 0
        }), 500


def verify_face_sequence(frame_data_list, kf):
    """Core verification logic: encoding match + geometry check."""
    if not kf:
        return {
            'verified': False, 'name': 'Unknown',
            'status': 'NO KNOWN FACES LOADED',
            'avg_geometry_score': 9999.0, 'frames_analyzed': 0
        }
    
    if not frame_data_list or len(frame_data_list) < 5:
        count = len(frame_data_list) if frame_data_list else 0
        return {
            'verified': False, 'name': 'Unknown',
            'status': f'INSUFFICIENT FRAMES ({count}/5)',
            'avg_geometry_score': 9999.0, 'frames_analyzed': count
        }
    
    # Step 1: Identify by encoding (majority vote)
    name_votes = {}
    for fd in frame_data_list:
        encoding = fd['encoding']
        matches = face_recognition.compare_faces(
            [d['encoding'] for d in kf.values()], encoding, tolerance=ENCODING_THRESHOLD
        )
        if True in matches:
            distances = face_recognition.face_distance(
                [d['encoding'] for d in kf.values()], encoding
            )
            best_idx = np.argmin(distances)
            if matches[best_idx]:
                name = list(kf.keys())[best_idx]
                name_votes[name] = name_votes.get(name, 0) + 1
    
    if not name_votes:
        return {
            'verified': False, 'name': 'Unknown',
            'status': 'NO ENCODING MATCH',
            'avg_geometry_score': 9999.0,
            'frames_analyzed': len(frame_data_list)
        }
    
    best_name = max(name_votes, key=name_votes.get)
    known_lm = kf[best_name]['landmarks']
    
    # Step 2: Geometry verification
    geo_scores = []
    for fd in frame_data_list:
        score = weighted_geometry_distance(fd['landmarks'], known_lm)
        geo_scores.append(score)
    
    avg_geo = sum(geo_scores) / len(geo_scores)
    geometry_pass = avg_geo < GEOMETRY_THRESHOLD
    
    return {
        'verified': bool(geometry_pass),
        'name': str(best_name),
        'status': 'GEOMETRY MATCH' if geometry_pass else 'FACE MISMATCH',
        'avg_geometry_score': float(avg_geo),
        'frames_analyzed': int(len(frame_data_list))
    }


if __name__ == "__main__":
    print("[API Server] Loading known faces...")
    known_faces = load_known_faces()
    print(f"[API Server] Loaded {len(known_faces)} known face(s).")
    print("[API Server] Starting on http://0.0.0.0:5050/verify")
    app.run(host="0.0.0.0", port=5050, debug=False)
