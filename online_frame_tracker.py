import cv2
import face_recognition
import numpy as np
import os

from online_geometry import (
    preprocess_frame,
    load_known_faces,
    verify_face_sequence,
    get_eye_aspect_ratio,
)

# --- Configuration ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
MAX_FRAMES = 30          # Max frames to collect before sending
MIN_FRAMES = 5           # Min frames required
PROCESS_EVERY_N = 5      # Process every N-th camera frame (for performance)
BLINK_THRESHOLD = 0.2    # Eye Aspect Ratio threshold for blink detection
BLINK_CONSEC_MIN = 1     # Min consecutive closed-eye frames for a valid blink
BLINK_CONSEC_MAX = 2     # Max consecutive closed-eye frames for a valid blink


# --- Liveness Detection (Client-side) ---
def check_liveness(ear_history):
    """
    Checks liveness by analyzing EAR history for valid blink patterns.
    A valid blink = 2-4 consecutive frames with EAR < BLINK_THRESHOLD.
    
    Args:
        ear_history: List of EAR values from tracked frames.
    
    Returns:
        tuple: (is_live: bool, blink_count: int)
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
    
    # Check if sequence ends with a valid blink
    if BLINK_CONSEC_MIN <= consecutive_closed <= BLINK_CONSEC_MAX:
        blink_count += 1
    
    return blink_count > 0, blink_count


def main():
    print("[FrameTracker] Loading known faces...")
    known_faces = load_known_faces()
    if not known_faces:
        print("[FrameTracker] WARNING: No known faces loaded. Verification will fail.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FrameTracker] Error: Could not open camera.")
        return

    print("[FrameTracker] Camera opened. Starting face tracking...")
    print("[FrameTracker] Please look at the camera and BLINK naturally.")
    print("[FrameTracker] Press 'q' to quit, 'r' to reset buffer.\n")

    frame_buffer = []   # List of {'encoding': ..., 'landmarks': ...}
    ear_history = []    # EAR values for liveness detection
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

                # Use the first detected face
                top, right, bottom, left = face_locations[0]
                encoding = encodings[0]
                landmarks = landmarks_list[0]
                last_face_box = (top, right, bottom, left)

                # Compute EAR for liveness
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

                # --- Check if buffer is full ---
                if collected >= MAX_FRAMES:
                    print("\n[FrameTracker] Buffer full. Running verification...")
                    
                    # 1. Liveness check (client-side)
                    is_live, blink_count = check_liveness(ear_history)
                    
                    # 2. Encoding + Geometry check (server-side)
                    geo_result = verify_face_sequence(frame_buffer, known_faces)
                    
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
            # Show result
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
            # Show collection progress
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
            # Reset
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
