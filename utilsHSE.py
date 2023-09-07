import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential, load_model,model_from_json
import csv
import torch
from utilsVerify import *
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

def convert_to_gpu_tensor(faces):
    # Check if a GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        # GPU is available, convert to GPU tensor
        gpu_faces = tf.convert_to_tensor(faces, dtype=tf.float32)
        return gpu_faces
    else:
        # No GPU available, return the input as is
        return faces


import concurrent.futures


def detect_bboxes(frame, confidence_threshold=0.9):
    # Detects bboxes of faces in one frame using MTCNN

    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    # If any faces detected, make sure they're confident!
    if bounding_boxes.shape[0] > 0: 
        bounding_boxes=bounding_boxes[probs>confidence_threshold]
    return bounding_boxes

# Letterbox function
def letterbox_image_np(image_np, desired_size):
    height, width = image_np.shape[:2]
    ratio = min(desired_size[0] / width, desired_size[1] / height)
    new_size = (round(width * ratio), round(height * ratio))
    image_np_resized = cv2.resize(image_np, new_size, interpolation=cv2.INTER_LINEAR)
    new_im_np = np.zeros((desired_size[1], desired_size[0], 3), dtype=np.uint8)
    top_left_y = (desired_size[1] - new_size[1]) // 2
    top_left_x = (desired_size[0] - new_size[0]) // 2
    new_im_np[top_left_y:top_left_y + new_size[1], top_left_x:top_left_x + new_size[0]] = image_np_resized
    return new_im_np

def process_frame(frame, INPUT_SIZE):
    bounding_boxes = detect_bboxes(frame)
    if bounding_boxes.shape[0] == 1: # take only frames w one face!
        box = bounding_boxes[0].astype(int) # take only first face
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        if not face_img.size: # check if face_img is empty
            return None, True
        
        face_img=letterbox_image_np(face_img, INPUT_SIZE)
        return face_img, False
    else:
        return None, True


def extract_faces_mtcnn(frames, INPUT_SIZE):
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame in frames:
            future = executor.submit(process_frame, frame, INPUT_SIZE)
            futures.append(future)
        for enum, future in enumerate(concurrent.futures.as_completed(futures)):
            inp, null = future.result()
            if not null:
                faces[enum, :, :, :] = inp
            else:
                is_null[enum] = 1

    return faces, is_null

def extract_faces_with_verify(frames, INPUT_SIZE, target_img_path):
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
    verification_indices = [] # Pool frames with >1 face and send to verification pipeline
    for enum, frame in enumerate(frames):
        bounding_boxes = detect_bboxes(frame)
        if bounding_boxes.shape[0] == 1: # frames with one face
            box = bounding_boxes[0].astype(int)
            x1,y1,x2,y2=box[0:4]    
            face_img=frame[y1:y2,x1:x2,:]
            
            if not face_img.size: # check if face_img is empty
                is_null[enum] = 1
            else:
                face_img=letterbox_image_np(face_img, INPUT_SIZE)
                faces[enum] = face_img 
        elif bounding_boxes.shape[0] > 1: # more than one face!
            verification_indices.append(enum)
            is_null[enum] = 1 # It's null for now, but will be valid if verified! 
        else: # zero faces
            is_null[enum] = 1
    if len(verification_indices) > 0: 
        verify_np_array = frames[verification_indices] 
        verify_results = verify_faces_np_data(target_img_path, verify_np_array)
        for _, row in verify_results.iterrows():
            idx = row['Index']
            real_index = verification_indices[int(idx)]
            x, y, w, h = int(row['Facial Box X']), int(row['Facial Box Y']), int(row['Facial Box W']), int(row['Facial Box H'])
            full_image = frames[real_index]
            face_img = full_image[y:y+h, x:x+w, :]
            face_img=letterbox_image_np(face_img, INPUT_SIZE)

            # # DEBUG ONLY: SAVE THE IMAGES, SHOWING BBOXES!
            # showing_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR) 
            # draw_bbox_and_save(showing_image, (x, y, w, h), os.path.abspath(f'outputs_Combined/Verification_Demo.mp4/frame_{real_index*6}.jpg'))

            faces[real_index] = face_img
            is_null[real_index] = 0 # it's been verified, so it is not null

    return faces, is_null

def draw_bbox_and_save(img, bbox, filepath):
    # Unpack the bounding box
    x, y, w, h = bbox

    # Draw the bounding box onto the image
    # The arguments here are the image, the top-left corner, the bottom-right corner,
    # the color, and the line width, respectively.
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the image to the specified filepath
    cv2.imwrite(filepath, img)


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
    
    # Make a modified array with (frame, timestamp, success) before emotions
    success_array = 1 - is_null
    modified_arr = np.concatenate((np.array(success_array).reshape(-1, 1), labels), axis=1)
    timestamps = [frame / fps for frame in frames]
    modified_arr = np.concatenate((frames, np.array(timestamps), modified_arr), axis=1)
    
    # Save the data to the CSV file, making sure to append and not write over!
    with open(save_path, 'a') as file:
        np.savetxt(file, modified_arr, delimiter=',', header='', footer='', comments='')



