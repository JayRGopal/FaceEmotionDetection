import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential, load_model,model_from_json
import csv
from utilsVerify import *
from utils import *
import random
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'

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
    if bounding_boxes is None:
        return np.array([]) # better to return empty array than None!
    if bounding_boxes.shape[0] > 0:
        bounding_boxes=bounding_boxes[probs>confidence_threshold]
    return bounding_boxes


def process_frame(frame, INPUT_SIZE):
    bounding_boxes = detect_bboxes(frame)
    if bounding_boxes.shape[0] == 1: # take only frames w one face!
        box = bounding_boxes[0].astype(int) # take only first face
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        if not face_img.size: # check if face_img is empty
            return None, True, []
        
        face_img=letterbox_image_np(face_img, INPUT_SIZE)
        return face_img, False, [x1,y1,x2-x1,y2-y1]
    else:
        return None, True, []


def extract_faces_mtcnn(frames, INPUT_SIZE, real_frame_numbers=[]):
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
    all_bboxes = [] # Save the bboxes!

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for frame in frames:
            future = executor.submit(process_frame, frame, INPUT_SIZE)
            futures.append(future)
        for enum, future in enumerate(concurrent.futures.as_completed(futures)):
            inp, null, bbox = future.result()
            if not null:
                faces[enum, :, :, :] = inp
                all_bboxes.append((real_frame_numbers[enum], bbox[0], bbox[1], bbox[2], bbox[3]))
            else:
                is_null[enum] = 1
                all_bboxes.append((real_frame_numbers[enum], 0, 0, 0, 0))
    all_bboxes = pd.DataFrame(all_bboxes, columns=['Index', 'Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H'])
    

    return faces, is_null, all_bboxes

def extract_faces_with_verify(frames, INPUT_SIZE, target_img_folder, partialVerify=False, verifyAll=False, distance_max=30, save_folder_path='', real_frame_numbers=[], saveProb=0.01):
    # Extracts faces using MTCNN from frames
    # Reshapes using letterbox to INPUT_SIZE
    # target_img_folder has the verification target images (JPEGs)
    # partialVerify: if this is true, we don't verify every frame with 2+ faces. We check if there's a face close to the last verified face
    # verifyAll: if this is true, 
    # distance_max for partialVerify. If nearest face is beyond distance_max, verification occurs again.
    # save_folder_path - only used for partialVerify to randomly save 1% of partially verified faces.
    # real_frame_numbers - only used for partialVerify. If a frame is saved, its number is also saved.
    # saveProb - probability that a partially verified frame is saved

    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
    verification_indices = [] # Pool frames with >1 face and send to verification pipeline
    verification_bboxes = []
    all_bboxes = [] # Save the bboxes!
    for enum, frame in enumerate(frames):
        bounding_boxes = detect_bboxes(frame)
        if bounding_boxes.shape[0] == 1: # frames with one face
            box = bounding_boxes[0].astype(int)
            x1,y1,x2,y2=box[0:4]    
            face_img=frame[y1:y2,x1:x2,:]
            
            if not face_img.size: # check if face_img is empty
                is_null[enum] = 1
                all_bboxes.append((real_frame_numbers[enum], 0, 0, 0, 0)) # add empty face to bboxes
            else:
                face_img=letterbox_image_np(face_img, INPUT_SIZE)
                faces[enum] = face_img 

                all_bboxes.append((real_frame_numbers[enum], x1, y1, x2 - x1, y2 - y1))
        elif bounding_boxes.shape[0] > 1: # more than one face!
            verification_indices.append(enum)
            verification_bboxes.append(bounding_boxes)
            is_null[enum] = 2 # It's null for now, but will be valid if verified! 
            all_bboxes.append((real_frame_numbers[enum], 0, 0, 0, 0)) # add empty face to bboxes
        else: # zero faces
            is_null[enum] = 1
            all_bboxes.append((real_frame_numbers[enum], 0, 0, 0, 0)) # add empty face to bboxes
    
    # Convert all_bboxes to a pd dataframe
    all_bboxes = pd.DataFrame(all_bboxes, columns=['Index', 'Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H'])
    
    if len(verification_indices) > 0: 
        verify_np_array = frames[verification_indices] 
        if partialVerify:
            verify_results = verify_partial_faces_np_data(target_img_folder, verify_np_array, verification_bboxes, distance_max=distance_max)
        else:
            verify_results = verify_faces_np_data(target_img_folder, verify_np_array)
        
        if verify_results.shape and verify_results.shape[0] > 0:
            df_copy = verify_results.copy()
            df_copy['Index'] = df_copy['Index'].apply(lambda idx: real_frame_numbers[verification_indices[int(idx)]])
            df_copy = df_copy[['Index', 'Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']]
            all_bboxes = all_bboxes[~all_bboxes['Index'].isin(df_copy['Index'])]
            merged_df = pd.concat([all_bboxes, df_copy])
            merged_df.sort_values(by='Index', inplace=True)
            all_bboxes = merged_df


        # DEBUG ONLY: SAVE THE FULL VERIFICATION BBOXES!
        # df_copy.to_csv(os.path.abspath(f'outputs_Combined/Verification_Demo.mp4/verify_bboxes.csv'), index=False)    
        

        for _, row in verify_results.iterrows():
            idx = row['Index']
            real_index = verification_indices[int(idx)]
            x, y, w, h = int(row['Facial Box X']), int(row['Facial Box Y']), int(row['Facial Box W']), int(row['Facial Box H'])
            if w > 0 and h > 0: # Avoid "empty" verifications
                full_image = frames[real_index]
                face_img = full_image[y:y+h, x:x+w, :]
                face_img=letterbox_image_np(face_img, INPUT_SIZE)

                if partialVerify and row['Partial Verify']:
                    savingThisFrame = random.random() < saveProb
                    if savingThisFrame:
                        showing_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(save_folder_path, f'partialVerify_{real_frame_numbers[real_index]}.jpg'), showing_face)

                # DEBUG ONLY: SAVE THE IMAGES, SHOWING BBOXES!
                # showing_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR) 
                # draw_bbox_and_save(showing_image, (x, y, w, h), os.path.abspath(f'outputs_Combined/Fallon_Kimmel_Demo.mp4/frame_{real_index*6}.jpg'))

                faces[real_index] = face_img
                is_null[real_index] = 0 # it's been verified, so it is not null

    return faces, is_null, all_bboxes


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

    # If empty, return empty
    if faces.shape[0] == 0:
        return np.array([])
    
    # Device
    if device_name != '' and '/device:GPU' in device_name:
        with tf.device('/device:GPU:0'):
            scores = model.predict(faces)
    else:
        scores = model.predict(faces)

    return scores

def csv_save_HSE(labels, is_null, frames, save_path, fps):
    if labels.shape[0] == 0: # 0 frames successfully found in whole batch (due to downsampling)!
        class_labels=['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        labels = np.zeros((frames.shape[0], 7))
    elif labels.shape[1] == 7: # 7 emotions - mobilenet
        class_labels=['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    else: 
        print(f'Unexpected shape of hse labels! {labels.shape}') 
        return
    if not(os.path.exists(save_path)): # Make a new file with the correct first rows
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            row1 = ['frame', 'timestamp', 'success'] + class_labels
            writer.writerow(row1)
    
    # Make a modified array with (frame, timestamp, success) before emotions
    success_array = 1 - is_null
    modified_arr = np.concatenate((np.array(success_array).reshape(-1, 1), labels), axis=1)
    frames_t = frames.astype(np.float32)
    timestamps = [frame / fps for frame in frames_t]
    if frames_t.shape[0] == 0:
        # No data to save, so let's move on!
        return
    else:
        modified_arr = np.concatenate((frames, np.array(timestamps), modified_arr), axis=1)
    
        # Save the data to the CSV file, making sure to append and not write over!
        with open(save_path, 'a') as file:
            np.savetxt(file, modified_arr, delimiter=',', header='', footer='', comments='')


def csv_save_bboxes(labels, is_null, frames, save_path, fps):
    if labels.shape[0] == 0: # 0 frames successfully found in whole batch (due to downsampling)!
        class_labels=['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']
        labels = np.zeros((frames.shape[0], 4))
    elif labels.shape[1] == 4: # 4 datapoints for bounding box
        class_labels=['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']
    else: 
        print(f'Unexpected shape of bboxes array! {labels.shape}') 
        return
    if not(os.path.exists(save_path)): # Make a new file with the correct first rows
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            row1 = ['frame', 'timestamp', 'success'] + class_labels
            writer.writerow(row1)
    
    # Make a modified array with (frame, timestamp, success) before emotions
    success_array = 1 - is_null
    modified_arr = np.concatenate((np.array(success_array).reshape(-1, 1), labels), axis=1)
    frames_t = frames.astype(np.float32)
    timestamps = [frame / fps for frame in frames_t]
    if frames_t.shape[0] == 0:
        # No data to save, so let's move on!
        return
    else:
        modified_arr = np.concatenate((frames, np.array(timestamps), modified_arr), axis=1)
    
        # Save the data to the CSV file, making sure to append and not write over!
        with open(save_path, 'a') as file:
            np.savetxt(file, modified_arr, delimiter=',', header='', footer='', comments='')



