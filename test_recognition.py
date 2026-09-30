import cv2
from face_recognition import recognize_face


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame, results = recognize_face(frame)


    # Print recognition result
    for result in results:

        print(
            result["name"],
            result["score"]
        )


    cv2.imshow(
        "Recognition Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows() 