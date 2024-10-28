from utilsHSE import *
import os
import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
import csv

# Main parameters
PAT_NOW = "S23_199"
VIDEO_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/MP4/{PAT_NOW}_MP4')
OUTPUT_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
MODEL_PATH = os.path.abspath('eyes_closed_models/shape_predictor_68_face_landmarks.dat')

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Define parameters
thresh = 0.25
frame_check = 20
FPS_EXTRACTING = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(MODEL_PATH)

# Landmark indices for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_video(video_path):
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_division = int(fps // FPS_EXTRACTING)
    frameNr = 0
    flag = 0
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frameNr % frame_division == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            success_flag = -2  # Initialize with -2 for "no faces detected" case
            drowsiness_detected = 0
            drowsiness_score = 0.0

            if len(subjects) == 1:  # Successful detection of one face
                shape = predict(gray, subjects[0])
                shape = face_utils.shape_to_np(shape)
                
                # Compute EAR
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                drowsiness_score = ear
                
                # Determine drowsiness
                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        drowsiness_detected = 1
                else:
                    flag = 0
                
                success_flag = 1  # One face detected successfully
            elif len(subjects) > 1:
                success_flag = -1  # Failure: More than one face detected

            # Save frame information
            timestamp = str(datetime.now())
            data.append([frameNr, timestamp, success_flag, drowsiness_detected, drowsiness_score])

        frameNr += 1
    
    cap.release()
    
    # Save to CSV using helper function
    output_file = os.path.join(OUTPUT_DIRECTORY, f"{os.path.basename(video_path)}", "outputs_drowsiness.csv")
    csv_save_drowsiness(data, output_file)

# Process all videos in the directory
for video_file in os.listdir(VIDEO_DIRECTORY):
    if video_file.endswith(".mp4"):
        process_video(os.path.join(VIDEO_DIRECTORY, video_file))

print("Processing complete for all videos.")
