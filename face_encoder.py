import face_recognition
import argparse
import pickle
import os
import cv2

def encode_face(image_path, output_name):
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        print("Error: No face detected in the image.")
        return

    # Get landmarks
    face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)

    if not face_landmarks_list:
        print("Error: Could not extract face landmarks.")
        return

    # Ensure output directory exists
    output_dir = "known_faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{output_name}.bin")

    # Save to file
    with open(output_path, "wb") as f:
        pickle.dump(face_landmarks_list, f)

    print(f"Success! Face geometry saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode face geometry from an image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("name", help="Name of the person (output filename).")
    args = parser.parse_args()

    encode_face(args.image, args.name)
