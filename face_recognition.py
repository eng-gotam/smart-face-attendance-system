import cv2
import pickle
import os
import numpy as np


# --------------------------------
# Model paths
# --------------------------------

FACE_DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

DATABASE_FILE = "face_database.pkl"


# --------------------------------
# Load models
# --------------------------------

face_detector = cv2.FaceDetectorYN.create(
    FACE_DETECTOR_MODEL,
    "",
    (640, 640)
)

face_recognizer = cv2.FaceRecognizerSF.create(
    FACE_RECOGNIZER_MODEL,
    ""
)


# --------------------------------
# Load database safely
# --------------------------------

database = {}

if os.path.exists(DATABASE_FILE):

    with open(DATABASE_FILE, "rb") as file:
        database = pickle.load(file)

    print("Face database loaded.")

else:

    print("Face database not found.")
    print("Please register faces before recognition.")


# --------------------------------
# Recognize face
# --------------------------------

def recognize_face(image):

    if not database:
        return "Unknown", 0.0

    image = cv2.resize(
        image,
        (640, 640)
    )

    face_detector.setInputSize(
        (640, 640)
    )

    _, faces = face_detector.detect(
        image
    )

    if faces is None:
        return "No Face", 0.0

    best_person = "Unknown"
    best_score = -1

    for face in faces:

        aligned_face = face_recognizer.alignCrop(
            image,
            face
        )

        feature = face_recognizer.feature(
            aligned_face
        )

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(1, -1)

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

                if score > best_score:

                    best_score = score
                    best_person = person


    # --------------------------------
    # Recognition threshold
    # --------------------------------

    THRESHOLD = 0.60

    if best_score >= THRESHOLD:

        return best_person, best_score

    return "Unknown", best_score
