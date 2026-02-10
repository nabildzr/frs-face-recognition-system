import cv2
import face_recognition
import numpy as np
import os
import base64
import requests
import json

# --- Configuration ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
API_SERVER_URL = "http://localhost:5050/verify"
MAX_FRAMES = 25
MIN_FRAMES = 5
PROCESS_EVERY_N = 5
BLINK_THRESHOLD = 0.2
BLINK_CONSEC_MIN = 1
BLINK_CONSEC_MAX = 2


# --- EAR Calculation (local) ---
def get_eye_aspect_ratio(eye_landmarks):
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (v1 + v2) / (2.0 * h)


# --- Preprocessing (local) ---
def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


# --- Liveness Detection (Client-side) ---
def check_liveness(ear_history):
    """
    Checks liveness by analyzing EAR history for valid blink patterns.
    A valid blink = 1-2 consecutive frames with EAR < BLINK_THRESHOLD.
    """
    blink_count = 0
    consecutive_closed = 0
    
    for ear in ear_history:
        if ear < BLINK_THRESHOLD:
            consecutive_closed += 1
        else:
            if BLINK_CONSEC_MIN <= consecutive_closed <= BLINK_CONSEC_MAX:
                blink_count += 1
            consecutive_closed = 0
    
    if BLINK_CONSEC_MIN <= consecutive_closed <= BLINK_CONSEC_MAX:
        blink_count += 1
    
    return blink_count > 0, blink_count


# --- Serialization ---
def serialize_frame_buffer(frame_buffer):
    """Convert frame buffer to JSON-serializable format (base64 encoding + landmarks)."""
    serialized = []
    for frame_data in frame_buffer:
        serialized.append({
            'encoding': base64.b64encode(frame_data['encoding'].tobytes()).decode('utf-8'),
            'landmarks': {
                key: [list(point) for point in points]
                for key, points in frame_data['landmarks'].items()
            },
        })
    return serialized


# --- Send to Server ---
def send_to_server(frame_buffer):
    """Send frame data to api_geometry.py server via HTTP POST."""
    try:
        payload = {
            'frames': serialize_frame_buffer(frame_buffer)
        }
        
        response = requests.post(
            API_SERVER_URL,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[FrameTracker] Server error: {response.status_code}")
            return {
                'verified': False, 'name': 'Unknown',
                'status': f'SERVER ERROR ({response.status_code})',
                'avg_geometry_score': float('inf'), 'frames_analyzed': 0
            }
    except requests.exceptions.ConnectionError:
        print("[FrameTracker] ERROR: Cannot connect to server. Is api_geometry.py running?")
        return {
            'verified': False, 'name': 'Unknown',
            'status': 'SERVER UNREACHABLE',
            'avg_geometry_score': float('inf'), 'frames_analyzed': 0
        }
    except Exception as e:
        print(f"[FrameTracker] Request error: {e}")
        return {
            'verified': False, 'name': 'Unknown',
            'status': f'REQUEST ERROR: {str(e)}',
            'avg_geometry_score': float('inf'), 'frames_analyzed': 0
        }


def main():
    print("[FrameTracker] Starting face tracker client...")
    print(f"[FrameTracker] Server URL: {API_SERVER_URL}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FrameTracker] Error: Could not open camera.")
        return

    print("[FrameTracker] Camera opened. Please look at the camera and BLINK naturally.")
    print("[FrameTracker] Press 'q' to quit, 'r' to reset buffer.\n")

    frame_buffer = []
    ear_history = []
    frame_count = 0
    verification_result = None
    last_face_box = None
    blink_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # --- Process every N-th frame ---
        if frame_count % PROCESS_EVERY_N == 0 and verification_result is None:
            processed = preprocess_frame(frame)
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb)

            if face_locations:
                encodings = face_recognition.face_encodings(rgb, face_locations)
                landmarks_list = face_recognition.face_landmarks(rgb, face_locations)

                top, right, bottom, left = face_locations[0]
                encoding = encodings[0]
                landmarks = landmarks_list[0]
                last_face_box = (top, right, bottom, left)

                # Compute EAR for liveness (local)
                left_ear = get_eye_aspect_ratio(landmarks['left_eye'])
                right_ear = get_eye_aspect_ratio(landmarks['right_eye'])
                avg_ear = (left_ear + right_ear) / 2.0

                # Store data
                frame_buffer.append({
                    'encoding': encoding,
                    'landmarks': landmarks,
                })
                ear_history.append(avg_ear)

                collected = len(frame_buffer)
                blink_status = "EYES CLOSED" if avg_ear < BLINK_THRESHOLD else "EYES OPEN"
                if avg_ear < BLINK_THRESHOLD:
                    blink_detected = True

                print(f"  [Frame {collected}/{MAX_FRAMES}] EAR: {avg_ear:.3f} | {blink_status}")

                # --- Buffer full -> verify ---
                if collected >= MAX_FRAMES:
                    print("\n[FrameTracker] Buffer full. Running verification...")
                    
                    # 1. Liveness check (client-side)
                    is_live, blink_count = check_liveness(ear_history)
                    
                    # 2. Encoding + Geometry check (HTTP to server)
                    print("[FrameTracker] Sending data to server...")
                    geo_result = send_to_server(frame_buffer)
                    
                    # 3. Combine results
                    final_verified = is_live and geo_result['verified']
                    
                    if final_verified:
                        final_status = "ACCESS GRANTED"
                    elif not is_live:
                        final_status = "LIVENESS FAILED - BLINK REQUIRED"
                    else:
                        final_status = geo_result['status']
                    
                    verification_result = {
                        'verified': final_verified,
                        'name': geo_result['name'],
                        'status': final_status,
                        'avg_geometry_score': geo_result['avg_geometry_score'],
                        'liveness': is_live,
                        'blink_count': blink_count,
                        'frames_analyzed': geo_result['frames_analyzed'],
                    }
                    _print_result(verification_result)

            else:
                last_face_box = None

        # --- Draw UI ---
        if verification_result is not None:
            if verification_result['verified']:
                color = (0, 255, 0)
                label = f"VERIFIED: {verification_result['name']} | {verification_result['status']}"
            else:
                color = (0, 0, 255)
                label = f"FAILED: {verification_result['status']}"

            cv2.putText(display_frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, "Press 'r' to retry or 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if last_face_box:
                top, right, bottom, left = last_face_box
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        else:
            collected = len(frame_buffer)
            progress = f"Collecting: {collected}/{MAX_FRAMES} frames"
            cv2.putText(display_frame, progress, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if blink_detected:
                cv2.putText(display_frame, "Blink Detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "Please BLINK naturally", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if last_face_box:
                top, right, bottom, left = last_face_box
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 2)

        cv2.imshow("Face Tracker", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_buffer.clear()
            ear_history.clear()
            verification_result = None
            last_face_box = None
            blink_detected = False
            print("\n[FrameTracker] Buffer reset. Starting new collection...")

    cap.release()
    cv2.destroyAllWindows()


def _print_result(result):
    print("=" * 50)
    print(f"  Name:            {result['name']}")
    print(f"  Verified:        {result['verified']}")
    print(f"  Status:          {result['status']}")
    print(f"  Geometry Score:  {result['avg_geometry_score']:.4f}")
    print(f"  Liveness:        {result['liveness']}")
    print(f"  Blink Count:     {result['blink_count']}")
    print(f"  Frames Analyzed: {result['frames_analyzed']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
