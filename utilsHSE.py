import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential, load_model,model_from_json
import csv

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


def mobilenet_preprocess_input(x,**kwargs):
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


def fpath_to_var(fpath):
  frame_bgr=cv2.imread(fpath)
  frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  return frame

def extract_faces_mtcnn(frames, INPUT_SIZE):
    imgProcessing=FacialImageProcessing(False)
    is_null = np.zeros(frames.shape[0])
    faces = np.zeros([frames.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3])
    for enum, frame in enumerate(frames):
      bounding_boxes, points = imgProcessing.detect_faces(frame)
      if bounding_boxes.shape[0] > 0:
          box = bounding_boxes[0].astype(np.int) # take only first face
          x1,y1,x2,y2=box[0:4]    
          face_img=frame[y1:y2,x1:x2,:]
          face_img=cv2.resize(face_img, INPUT_SIZE)
          inp=face_img.astype(np.float32)
          faces[enum, :, :, :] = inp
      else:
          is_null[enum] = 1

    return faces, is_null

def hse_preds(faces, model, model_type='mobilenet_7.h5'):
  if model_type == 'mobilenet_7.h5':
      preprocessing_function=mobilenet_preprocess_input
  faces = preprocessing_function(faces)
  scores=model.predict(faces)
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



