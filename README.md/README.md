# Smart Face Attendance System

A real-time face recognition-based attendance system built with Python, OpenCV, YuNet, SFace, and Streamlit.

The system detects faces through a webcam, recognizes registered individuals using face embeddings, and automatically records their attendance in a CSV file.

## Features

- Real-time face detection
- Face recognition using SFace
- Face detection using YuNet
- Cosine similarity-based matching
- Automatic attendance recording
- Prevents duplicate attendance for the same person on the same day
- Streamlit web interface
- Live attendance monitoring
- Attendance CSV download
- Supports multiple registered people

## Technologies Used

- Python
- OpenCV
- YuNet
- SFace
- NumPy
- Pandas
- Streamlit

## System Architecture

```text
Webcam
   ↓
Face Detection (YuNet)
   ↓
Face Alignment
   ↓
Feature Extraction (SFace)
   ↓
Cosine Similarity
   ↓
Face Recognition
   ↓
Attendance Recording
   ↓
CSV File
Project Structure
Smart Face Attendance System/
│
├── app.py
├── face_recognition.py
├── test_recognition.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── face_detection_yunet_2023mar.onnx
    └── face_recognition_sface_2021dec.onnx
How It Works
The webcam captures a live video frame.
YuNet detects faces in the frame.
The detected face is aligned for recognition.
SFace extracts facial features.
The extracted features are compared with the registered face database using cosine similarity.
If the similarity score passes the recognition threshold, the person is identified.
The system records the person's name and attendance date/time.
Duplicate attendance for the same person on the same day is prevented.
Installation

Clone the repository:

git clone https://github.com/eng-gotam/smart-face-attendance-system.git

Move into the project directory:

cd smart-face-attendance-system

Install dependencies:

pip install -r requirements.txt
Running the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

Face Database

The face database is generated locally from registered users' face images.

For privacy and security reasons, real face images and the generated face_database.pkl file are excluded from this repository using .gitignore.

Attendance Data

Attendance records are stored locally in:

attendance.csv

This file is also excluded from the GitHub repository to avoid exposing personal attendance information.

Recognition

The system currently uses cosine similarity to compare live facial features with stored face embeddings.

The recognition threshold is:

THRESHOLD = 0.60

This value can be adjusted according to testing conditions and recognition requirements.

Future Improvements
Add user registration through the UI
Add admin authentication
Add attendance history and analytics
Add date-based attendance filtering
Add database support
Improve recognition under different lighting conditions
Add deployment-ready camera handling
Add multiple camera support
Author

Gotam Kumar

BS Artificial Intelligence Student
Sindh Madressatul Islam University, Karachi

GitHub: https://github.com/eng-gotam



