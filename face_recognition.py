import cv2
import pickle
import os
import base64
import numpy as np
import streamlit as st


# --------------------------------
# Load models
# --------------------------------

face_detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (640, 640)
)

face_recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)


# --------------------------------
# Load face database
# --------------------------------

def load_face_database():

    # Local development
    if os.path.exists("face_database.pkl"):

        with open("face_database.pkl", "rb") as file:
            database = pickle.load(file)

        print("Face database loaded from local file.")

        return database


    # Streamlit Cloud
    if "FACE_DATABASE" in st.secrets:

        encoded_database = st.secrets["FACE_DATABASE"]

        database_bytes = base64.b64decode(
            encoded_database
        )

        database = pickle.loads(
            database_bytes
        )

        print("Face database loaded from Streamlit Secrets.")

        return database


    print("Face database not found.")

    return {}


database = load_face_database()


print("Face database loaded.")

for person, features in database.items():

    print(
        person,
        "->",
        len(features),
        "features"
    )


# --------------------------------
# Recognition function
# --------------------------------

def recognize_face(frame):

    # Resize frame
    frame = cv2.resize(
        frame,
        (640, 640)
    )

    # Set detector input size
    face_detector.setInputSize(
        (640, 640)
    )

    # Detect faces
    _, faces = face_detector.detect(
        frame
    )


    # --------------------------------
    # No face detected
    # --------------------------------

    if faces is None:

        return frame, "No Face", 0.0


    # --------------------------------
    # Default recognition result
    # --------------------------------

    best_person = "Unknown"
    best_score = -1.0


    # --------------------------------
    # Process every detected face
    # --------------------------------

    for face in faces:

        # Align face
        aligned_face = face_recognizer.alignCrop(
            frame,
            face
        )

        # Extract face feature
        feature = face_recognizer.feature(
            aligned_face
        )

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(1, -1)


        # --------------------------------
        # Compare with face database
        # --------------------------------

        current_person = "Unknown"
        current_score = -1.0


        for person, stored_features in database.items():

            for stored_feature in stored_features:

                stored_feature = np.asarray(
                    stored_feature,
                    dtype=np.float32
                ).reshape(1, -1)


                score = face_recognizer.match(
                    feature,
                    stored_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )


                # Keep highest similarity
                if score > current_score:

                    current_score = score
                    current_person = person


        # --------------------------------
        # Recognition threshold
        # --------------------------------

        THRESHOLD = 0.60


        if current_score >= THRESHOLD:

            best_person = current_person
            best_score = current_score

        else:

            best_person = "Unknown"
            best_score = current_score


        # --------------------------------
        # Draw bounding box
        # --------------------------------

        x, y, w, h = face[:4].astype(int)


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Draw recognition result
        # --------------------------------

        cv2.putText(
            frame,
            f"{best_person} ({best_score:.2f})",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # --------------------------------
    # Return processed frame
    # --------------------------------

    return frame, best_person, best_score

    
