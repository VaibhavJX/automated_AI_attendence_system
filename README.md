# Automated AI Attendance System

This project uses InsightFace to recognize registered students from a classroom
camera stream and record attendance in a CSV file.

## Classroom scanner

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Register each student with a webcam. This creates or updates
   `student_database.pkl`:

   ```bash
   python register_student.py "Student Name"
   ```

   Keep one face in view until the default five samples are captured. Use
   `--camera 1` for another webcam or `--samples 10` for more samples. Press
   `q` to cancel.
3. Alternatively, register students through a local web page:

   ```bash
   python local_registration_web.py
   ```

   Open `http://127.0.0.1:5000` in a browser, allow camera access, enter the
   student's name and roll number, capture a photo, and submit it.
4. Start the scanner with a remote stream or local webcam:

   ```bash
   python attendance_scanner.py --source "http://172.51.151.66:8080/video"
   ```

   If that remote stream is unavailable, the script automatically falls back to the
   local webcam. You can also force a specific webcam:

   ```bash
   python attendance_scanner.py --source 0
   python attendance_scanner.py --source webcam --camera-index 0
   ```

Recognized students are recorded once per run in `classroom_attendance.csv`,
including their roll number. Recognized students from older databases without
roll numbers remain supported.
Press `q` in the video window to stop scanning.
