"""Local web application for registering students with a captured photo."""

import os
import pickle
import base64

import cv2
import insightface
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from insightface.app import FaceAnalysis


DATABASE_FILE = "student_database.pkl"
app = Flask(__name__)
face_app = None

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Register Student</title>
  <style>
    body { font-family: sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
    video, canvas { width: 100%; max-width: 520px; border-radius: 8px; background: #222; }
    canvas { display: none; }
    input, button { box-sizing: border-box; font-size: 1rem; margin-top: .7rem; padding: .7rem; width: 100%; }
    button { cursor: pointer; }
    #message { min-height: 1.5rem; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>Register Student</h1>
  <p>Enter the student details, then capture one clear face photo.</p>
  <form id="registration-form">
    <input id="name" name="name" placeholder="Student name" required>
    <input id="roll_number" name="roll_number" placeholder="Roll number" required>
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas"></canvas>
    <button type="button" id="capture">Capture photo</button>
    <button type="submit">Register student</button>
  </form>
  <div id="message" role="status"></div>
  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const message = document.getElementById("message");
    let photo = null;

    navigator.mediaDevices.getUserMedia({video: true})
      .then(stream => { video.srcObject = stream; })
      .catch(error => { message.textContent = "Camera access failed: " + error.message; });

    document.getElementById("capture").onclick = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      photo = canvas.toDataURL("image/jpeg", 0.9);
      message.textContent = "Photo captured. Submit to register.";
    };

    document.getElementById("registration-form").onsubmit = async event => {
      event.preventDefault();
      if (!photo) { message.textContent = "Capture a photo first."; return; }
      message.textContent = "Registering...";
      const response = await fetch("/register", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          name: document.getElementById("name").value,
          roll_number: document.getElementById("roll_number").value,
          image: photo
        })
      });
      const result = await response.json();
      message.textContent = result.message;
      if (response.ok) {
        document.getElementById("registration-form").reset();
        photo = null;
      }
    };
  </script>
</body>
</html>
"""


def get_face_app():
    global face_app
    if face_app is None:
        print("Initializing face model...")
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return [], [], []

    with open(DATABASE_FILE, "rb") as database_file:
        data = pickle.load(database_file)
    names = data.get("names", [])
    embeddings = data.get("embeddings", [])
    roll_numbers = data.get("roll_numbers", [""] * len(names))
    if len(names) != len(embeddings) or len(roll_numbers) != len(names):
        raise ValueError("Student database has mismatched profile fields.")
    return names, roll_numbers, embeddings


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    roll_number = str(payload.get("roll_number", "")).strip()
    image_data = str(payload.get("image", ""))

    if not name or not roll_number or "," in roll_number:
        return jsonify(message="Name and a valid roll number are required."), 400
    if not image_data.startswith("data:image/"):
        return jsonify(message="A captured image is required."), 400

    try:
        encoded_image = image_data.split(",", 1)[1]
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded_image), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("The captured image could not be decoded.")

        faces = get_face_app().get(image)
        if len(faces) != 1:
            return jsonify(message="Ensure exactly one face is visible."), 400

        names, roll_numbers, embeddings = load_database()
        if name in names or roll_number in roll_numbers:
            return jsonify(message="That name or roll number is already registered."), 409

        names.append(name)
        roll_numbers.append(roll_number)
        embeddings.append(faces[0].embedding.astype(np.float32))
        with open(DATABASE_FILE, "wb") as database_file:
            pickle.dump(
                {
                    "names": names,
                    "roll_numbers": roll_numbers,
                    "embeddings": embeddings,
                },
                database_file,
            )
        return jsonify(message=f"Registered {name} (roll {roll_number}).")
    except (ValueError, OSError) as error:
        return jsonify(message=str(error)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
