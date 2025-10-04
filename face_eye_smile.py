import cv2

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier("C:/Users/91600/OneDrive/Desktop/c c++/live_face_detection_project/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/91600/OneDrive/Desktop/c c++/live_face_detection_project/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("C:/Users/91600/OneDrive/Desktop/c c++/live_face_detection_project/haarcascade_smile.xml")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define region of interest (ROI) for eyes/smile within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Optional: draw small rectangles around eyes
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Detect smile inside the face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=25)
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Smart Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()