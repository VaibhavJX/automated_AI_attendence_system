"""Scan a classroom video stream and record recognized students."""

import argparse
import csv
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis


EMBEDDINGS_FILE = "student_database.pkl"
CSV_FILE = "classroom_attendance.csv"
DEFAULT_RTSP_URL = "http://172.51.151.66:8080/video"
PROCESS_EVERY_N_FRAMES = 5
SIMILARITY_THRESHOLD = 0.40
LOCATION = "Classroom 101"


def parse_args():
    """Parse CLI options for the camera source."""
    parser = argparse.ArgumentParser(
        description="Run the AI attendance scanner using a remote stream or local webcam."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_RTSP_URL,
        help=(
            "Video source. Use a URL like http://host:8080/video, a local camera "
            "index like 0, or 'webcam' for the default local camera."
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Local webcam index to use if the primary source is unavailable.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to the local webcam when the configured stream fails.",
    )
    return parser.parse_args()


def open_video_capture(source, camera_index, fallback_enabled=True):
    """Open a video stream from a URL or the local webcam."""
    normalized_source = str(source).strip()

    if normalized_source.lower() in {"webcam", "local", "camera"}:
        normalized_source = str(camera_index)

    if normalized_source.isdigit():
        video_capture = cv2.VideoCapture(int(normalized_source))
    elif normalized_source.startswith(("http://", "https://", "rtsp://")):
        video_capture = cv2.VideoCapture(normalized_source)
        if video_capture.isOpened():
            return video_capture, normalized_source
        if fallback_enabled:
            print(
                f"Remote stream unavailable: {normalized_source}. "
                f"Falling back to local camera {camera_index}."
            )
            fallback_capture = cv2.VideoCapture(camera_index)
            if fallback_capture.isOpened():
                return fallback_capture, f"camera:{camera_index}"
        raise RuntimeError("Could not connect to camera stream.")
    else:
        video_capture = cv2.VideoCapture(normalized_source)

    if not video_capture.isOpened() and fallback_enabled:
        print(
            f"Could not open source '{normalized_source}'. "
            f"Falling back to local camera {camera_index}."
        )
        video_capture = cv2.VideoCapture(camera_index)

    if not video_capture.isOpened():
        raise RuntimeError("Could not connect to camera stream.")

    return video_capture, normalized_source


def load_profiles():
    """Load registered names and embeddings from the profile database."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(
            "No student database found! "
            "Please run 'register_student.py' first."
        )
        return [], np.empty((0, 0), dtype=np.float32)

    with open(EMBEDDINGS_FILE, "rb") as profiles_file:
        data = pickle.load(profiles_file)

    names = data.get("names", [])
    roll_numbers = data.get("roll_numbers", [""] * len(names))
    embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
    if embeddings.ndim == 1 and embeddings.size:
        embeddings = embeddings.reshape(1, -1)

    if len(names) != len(embeddings) or len(roll_numbers) != len(names):
        raise ValueError(
            "Student database contains mismatched profile fields."
        )

    print(f"Loaded {len(names)} registered student profiles.")
    return names, roll_numbers, embeddings


def ensure_attendance_file():
    """Create the attendance CSV with its header when it does not exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as attendance_file:
            csv.writer(attendance_file).writerow(
            ["Timestamp", "Roll Number", "Student Name", "Location", "Status"]
            )


def match_name(embedding, known_names, known_embeddings):
    """Return the best registered name when it exceeds the similarity threshold."""
    if not known_names:
        return "Unknown"

    embedding_norm = np.linalg.norm(embedding)
    known_norms = np.linalg.norm(known_embeddings, axis=1)
    similarities = np.dot(known_embeddings, embedding) / (
        known_norms * embedding_norm + 1e-10
    )
    best_match_idx = int(np.argmax(similarities))

    if similarities[best_match_idx] > SIMILARITY_THRESHOLD:
        return known_names[best_match_idx]
    return "Unknown"


def record_attendance(name, roll_number, present_students):
    """Record a recognized student once per scan."""
    if name == "Unknown" or name in present_students:
        return

    present_students.add(name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"MATCH FOUND: {name} detected at {timestamp}")

    with open(CSV_FILE, "a", newline="") as attendance_file:
        csv.writer(attendance_file).writerow(
            [timestamp, roll_number, name, LOCATION, "Present"]
        )


def draw_face(frame, bbox, name):
    """Draw a face bounding box and name label on the current frame."""
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    label_top = max(0, bbox[1] - 25)
    cv2.rectangle(
        frame,
        (bbox[0], label_top),
        (bbox[0] + 150, bbox[1]),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        name,
        (bbox[0] + 5, max(17, bbox[1] - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main():
    args = parse_args()
    print("Initializing AI Attendance System...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    known_names, roll_numbers, known_embeddings = load_profiles()
    ensure_attendance_file()

    video_capture, selected_source = open_video_capture(
        args.source,
        args.camera_index,
        fallback_enabled=not args.no_fallback,
    )
    print(f"Using camera source: {selected_source}")

    present_students = set()
    frame_count = 0
    current_boxes = []
    current_names = []

    print("Scanning classroom... Press 'q' to stop.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Video stream interrupted.")
                break

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                current_boxes = []
                current_names = []

                for face in app.get(frame):
                    bbox = face.bbox.astype(int)
                    name = match_name(
                        face.embedding, known_names, known_embeddings
                    )
                    student_index = (
                        known_names.index(name) if name != "Unknown" else -1
                    )
                    roll_number = (
                        roll_numbers[student_index] if student_index >= 0 else ""
                    )
                    current_boxes.append(bbox)
                    current_names.append(name)
                    record_attendance(name, roll_number, present_students)

            for bbox, name in zip(current_boxes, current_names):
                draw_face(frame, bbox, name)

            cv2.putText(
                frame,
                f"Present: {len(present_students)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Classroom Mass Scan - InsightFace", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    print(f"Scan completed. Total students present: {len(present_students)}")


if __name__ == "__main__":
    main()
