"""Register a student's face embedding in the local profile database."""

import argparse
import os
import pickle

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis


EMBEDDINGS_FILE = "student_database.pkl"


def load_database(path):
    if not os.path.exists(path):
        return [], []

    with open(path, "rb") as database_file:
        data = pickle.load(database_file)

    names = data.get("names", [])
    embeddings = data.get("embeddings", [])
    if len(names) != len(embeddings):
        raise ValueError("The existing database has mismatched names and embeddings.")
    return names, embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register one student's face in the attendance database."
    )
    parser.add_argument("name", help="Student name to register")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of face samples to average (default: 5)",
    )
    parser.add_argument(
        "--database",
        default=EMBEDDINGS_FILE,
        help=f"Database path (default: {EMBEDDINGS_FILE})",
    )
    args = parser.parse_args()
    if not args.name.strip():
        parser.error("name cannot be empty")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    return args


def main():
    args = parse_args()
    name = args.name.strip()
    names, embeddings = load_database(args.database)

    if name in names:
        raise ValueError(
            f"{name!r} is already registered. Remove the existing profile first."
        )

    print("Initializing face model...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera device {args.camera}.")

    captured_embeddings = []
    print("Look at the camera. Press 'q' to cancel.")

    try:
        while len(captured_embeddings) < args.samples:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            faces = app.get(frame)
            if len(faces) == 1:
                face = faces[0]
                captured_embeddings.append(face.embedding.astype(np.float32))
                message = (
                    f"Captured {len(captured_embeddings)}/{args.samples}"
                )
                color = (0, 255, 0)
                bbox = face.bbox.astype(int)
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    color,
                    2,
                )
            else:
                message = "Show exactly one face to the camera"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.imshow("Register Student", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Registration cancelled.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()

    embedding = np.mean(np.asarray(captured_embeddings), axis=0)
    embedding /= np.linalg.norm(embedding) + 1e-10
    names.append(name)
    embeddings.append(embedding)

    with open(args.database, "wb") as database_file:
        pickle.dump({"names": names, "embeddings": embeddings}, database_file)

    print(f"Registered {name!r} in {args.database}.")
    print(f"Total registered students: {len(names)}")


if __name__ == "__main__":
    main()
