import cv2 as cv
import numpy as np

# Load face cascade and eye cascade
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Capture video from webcam with DirectShow backend
video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Read all frames from webcam
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.flip(frame, 1)  # Mirror the video feed
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv.destroyAllWindows()
