import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential, load_model,model_from_json
import csv
import torch
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess=tf.compat.v1.Session(config=config)
# set_session(sess)

from HSE_facial_analysis import FacialImageProcessing


def get_emotion_predictor(MODEL_NOW):
    MODEL_PATH = os.path.join(os.path.abspath('HSE_models'), 
                            'affectnet_emotions', MODEL_NOW)
    model=load_model(MODEL_PATH)
    return model


import concurrent.futures

def process_frame(frame, imgProcessing, INPUT_SIZE):
    bounding_boxes, points = imgProcessing.detect_faces(frame)
    if bounding_boxes.shape[0] == 1: # take only frames w one face!
        box = bounding_boxes[0].astype(np.uint8) # take only first face
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        if not face_img.size: # check if face_img is empty
            return None, True
        
        face_img=cv2.resize(face_img, INPUT_SIZE)
        inp=face_img.astype(np.float32)
        return inp, False
    else:
        return None, True

def extract_faces_mtcnn(frames, INPUT_SIZE):
    imgProcessing=FacialImageProcessing(False)
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3])
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame in frames:
            future = executor.submit(process_frame, frame, imgProcessing, INPUT_SIZE)
            futures.append(future)
        for enum, future in enumerate(concurrent.futures.as_completed(futures)):
            inp, null = future.result()
            if not null:
                faces[enum, :, :, :] = inp
            else:
                is_null[enum] = 1

    return faces, is_null


def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    bounding_boxes=bounding_boxes[probs>0.9]
    return bounding_boxes

def process_frame_new(frame, INPUT_SIZE):
    bounding_boxes = detect_face(frame)
    if bounding_boxes.shape[0] == 1: # take only frames w one face!
        box = bounding_boxes[0].astype(int) # take only first face
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        if not face_img.size: # check if face_img is empty
            return None, True
        
        face_img=cv2.resize(face_img, INPUT_SIZE)
        return face_img, False
    else:
        return None, True


def extract_faces_mtcnn_new(frames, INPUT_SIZE):
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame in frames:
            future = executor.submit(process_frame_new, frame, INPUT_SIZE)
            futures.append(future)
        for enum, future in enumerate(concurrent.futures.as_completed(futures)):
            inp, null = future.result()
            if not null:
                faces[enum, :, :, :] = inp
            else:
                is_null[enum] = 1

    return faces, is_null

def hse_preds(faces, model, model_type='mobilenet_7.h5'):
    
    # Check if a GPU is available and use it if possible
    device_name = tf.test.gpu_device_name()
    if device_name != '' and '/device:GPU' in device_name:
        with tf.device('/device:GPU:0'):
            scores = model.predict(faces)
            scores = scores.cpu()
    else:
        scores = model.predict(faces)

    return scores

def csv_save_HSE(labels, is_null, frames, save_path, fps):
    if labels.shape[1] == 7: # 7 emotions - mobilenet
        class_labels=['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    else: 
        print(f'Unexpected shape of labels! {labels.shape}') 
        return
    if not(os.path.exists(save_path)): # Make a new file with the correct first rows
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            row1 = ['frame', 'timestamp', 'success'] + class_labels
            writer.writerow(row1)
    
    # Make a modified array with (frame, timestamp, success) before AUs
    success_array = 1 - is_null
    modified_arr = np.concatenate((np.array(success_array).reshape(-1, 1), labels), axis=1)
    frame_nums = frames
    timestamps = [frame / fps for frame in frame_nums]
    modified_arr = np.concatenate((frames, np.array(timestamps), modified_arr), axis=1)
    
    # Save the data to the CSV file, making sure to append and not write over!
    with open(save_path, 'a') as file:
        np.savetxt(file, modified_arr, delimiter=',', header='', footer='', comments='')



