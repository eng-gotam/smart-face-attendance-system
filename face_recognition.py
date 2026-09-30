
import cv2
import pickle
import numpy as np


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

with open("face_database.pkl", "rb") as file:
    database = pickle.load(file)


# --------------------------------
# Recognition threshold
# --------------------------------

THRESHOLD = 0.60


# --------------------------------
# Recognize face
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
    _, faces = face_detector.detect(frame)

    results = []

    if faces is None:
        return frame, results


    # --------------------------------
    # Process every detected face
    # --------------------------------

    for face in faces:

        # Face coordinates
        x, y, w, h = face[:4].astype(int)


        # --------------------------------
        # Align face
        # --------------------------------

        aligned_face = face_recognizer.alignCrop(
            frame,
            face
        )


        # --------------------------------
        # Extract feature
        # --------------------------------

        feature = face_recognizer.feature(
            aligned_face
        )

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        feature = feature.reshape(
            1, -1
        )


        # --------------------------------
        # Find best match
        # --------------------------------

        best_person = "Unknown"
        best_score = -1


        for person, stored_features in database.items():

            for stored_feature in stored_features:

                stored_feature = np.asarray(
                    stored_feature,
                    dtype=np.float32
                )

                stored_feature = stored_feature.reshape(
                    1, -1
                )


                score = face_recognizer.match(
                    feature,
                    stored_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )


                if score > best_score:

                    best_score = score
                    best_person = person


        # --------------------------------
        # Recognition decision
        # --------------------------------

        if best_score >= THRESHOLD:

            name = best_person

        else:

            name = "Unknown"


        # --------------------------------
        # Draw bounding box
        # --------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Display name + score
        # --------------------------------

        cv2.putText(
            frame,
            f"{name} ({best_score:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Store result
        # --------------------------------

        results.append({
            "name": name,
            "score": float(best_score),
            "box": (x, y, w, h)
        })


    return frame, results
