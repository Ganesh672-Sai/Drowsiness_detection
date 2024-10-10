from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
COUNTER = 0

# Load face cascade
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# This function calculates and returns eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2 * C)

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Load Caffe models for age and gender
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Define age and gender class labels (these should match the labels used when training the models)
age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
gender_labels = ['Male', 'Female']

# Create the Tkinter window
root = tk.Tk()
root.title("Drowsiness Detection")

# Create a canvas for video display
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create control buttons
def start_video():
    global video_active
    video_active = True
    update_frame()

def stop_video():
    global video_active
    video_active = False

start_button = ttk.Button(root, text="Start", command=start_video)
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = ttk.Button(root, text="Stop", command=stop_video)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

video_active = False

def update_frame():
    global COUNTER, video_active

    if video_active:
        ret, frame = video_capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0)
        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)
            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                pygame.mixer.music.stop()
                COUNTER = 0

            (x, y, w, h) = (shape[0][0], shape[0][1], shape[16][0] - shape[0][0], shape[8][1] - shape[0][1])
            face_img = frame[y:y + h, x:x + w]
            face_img_blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.8, 87.6, 114.0))

            # Predict age
            age_net.setInput(face_img_blob)
            age_preds = age_net.forward()
            age = age_labels[age_preds[0].argmax()]

            # Predict gender
            gender_net.setInput(face_img_blob)
            gender_preds = gender_net.forward()
            gender = gender_labels[gender_preds[0].argmax()]

            # Display the age and gender on the frame
            cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to ImageTk format and display it on the canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk  # Keep a reference to avoid garbage collection

        root.after(10, update_frame)  # Call this function again after 10 ms

# Start webcam video capture
video_capture = cv2.VideoCapture(0)
time.sleep(2)

# Start the Tkinter main loop
root.mainloop()

video_capture.release()
cv2.destroyAllWindows()