import numpy as np
import torch
import tensorflow as tf
import cv2
import os
import csv
import math
from torchvision import transforms
import random


# Device
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
  torch.cuda.empty_cache()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Hyperparameters - Post-Processing for OpenGraphAU
EMA_ALPHA = 0.9
SMA_WINDOW_SIZE = 10

"""

# Video Downsampling (NOT USING)

"""

from moviepy.editor import VideoFileClip

def downsample_vid(input_path, output_path, new_fps):
    # Load the original video file
    clip = VideoFileClip(input_path)

    # Set the desired frame rate for the subsampled video
    frame_rate = new_fps

    # Subsample the video to the desired frame rate
    subsampled_clip = clip.set_fps(frame_rate)

    # Save the subsampled video to a new file with optimized compression settings
    subsampled_clip.write_videofile(output_path)

    return


"""

# General Pipeline Utils

"""

def get_valid_vids(video_dir, save_path_folder_function):
    """
    Takes in a path to the input video directory
    And a function that takes in a given input video path and outputs the path to its save folder

    Outputs a list of all unprocessed videos in the input directory

    """

    all_files = [vid for vid in os.listdir(video_dir) if vid[0:1] != '.']

    valid_files = [vid for vid in all_files if os.path.isfile(os.path.join(video_dir, vid))]
    valid_videos = [vid for vid in valid_files if vid.endswith('.mp4')] 
    unprocessed_videos = [vid for vid in valid_videos if not(os.path.exists(save_path_folder_function(vid)))]

    return unprocessed_videos 



"""

# Video to Image Frames

"""

def extract_images(path, start_frame, num_to_extract, method='torch', fps=5): 
    capture = cv2.VideoCapture(path)

    ims = []
    frameNr = 0
    real_fps = math.ceil(capture.get(cv2.CAP_PROP_FPS))
    frame_division = real_fps // fps
    while frameNr < (start_frame + num_to_extract):
        success, frame = capture.read()
        if success:
            if frameNr >= start_frame:
                if frameNr % frame_division == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ims.append(frame)
        else:
            break
        frameNr = frameNr+1
    capture.release()

    ims = np.array(ims)
    if ims.shape[0] == 0: # if we're out of images
        return (ims, ims)
    if method == 'torch':
        im_test = torch.tensor(np.resize(ims[0:ims.shape[0]], (ims.shape[0], 3, 244, 244))).type(torch.float32)
    elif method == 'tensorflow':
        im_test = tf.constant(ims[0:ims.shape[0]], dtype=tf.float32)
        im_test = tf.image.resize(im_test, [224, 224])
    else:
       print('ERROR! Method is not torch or tensorflow!')
       return

    return (ims, im_test)



"""

# OpenGraphAU Preprocessing

"""


class ImagePreprocessor(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            self.normalize
        ])

    def preprocess(self, img_batch):
        preprocessed_images = []
        for img_tensor in img_batch:
            # Convert the uint8 tensor to a PIL image
            img_tensor = self.transform(img_tensor)

            # Convert the PIL image to a float tensor in the range [0, 1]
            #img_tensor = transforms.ToTensor()(img_pil)

            # Append the preprocessed image tensor to the list
            preprocessed_images.append(img_tensor)

        # Stack the preprocessed images to form the batch tensor
        preprocessed_batch = torch.stack(preprocessed_images)

        return preprocessed_batch


"""

# Face Detection (MTCNN)

"""



# See utilsHSE.py for the mtcnn detector. This is a helper function

# def mtcnn_to_torch(faces):
#     faces_real = torch.tensor(faces).float()
#     faces_real = torch.swapaxes(faces_real, 1, 3) / 255
#     return faces_real


def mtcnn_to_torch(faces):
    faces_real = torch.tensor(faces, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return faces_real


# Letterbox function
def letterbox_image_np(image_np, desired_size):
    height, width = image_np.shape[:2]
    if height == 0 or width == 0:
        new_im_np = np.zeros([desired_size[0], desired_size[1], 3], dtype=np.uint8)
    else:
        ratio = min(desired_size[0] / width, desired_size[1] / height)
        new_size = (round(width * ratio), round(height * ratio))
        image_np_resized = cv2.resize(image_np, new_size, interpolation=cv2.INTER_LINEAR)
        new_im_np = np.zeros((desired_size[1], desired_size[0], 3), dtype=np.uint8)
        top_left_y = (desired_size[1] - new_size[1]) // 2
        top_left_x = (desired_size[0] - new_size[0]) // 2
        new_im_np[top_left_y:top_left_y + new_size[1], top_left_x:top_left_x + new_size[0]] = image_np_resized
    return new_im_np

"""

# Face Detection (RetinaFace)

# We are NOT currently using this. 
# See utilsHSE.py for the MTCNN face extraction we are using.

"""

from retinaface import RetinaFace

def detect_extract_faces(ims, face_shape=(224, 224)):
  NUM_TO_EXTRACT = ims.shape[0]
  is_null = np.zeros(ims.shape[0], dtype=np.uint8)
  faces = np.zeros((NUM_TO_EXTRACT, face_shape[0], face_shape[1], 3), dtype=np.float32)
  for i in range(NUM_TO_EXTRACT):
    print(f'Extracting Face {i+1} / {NUM_TO_EXTRACT}')
    one_face = RetinaFace.extract_faces(ims[i])
    if len(one_face) > 0:
      one_face = np.array(one_face[0])
      dims_fixed = letterbox_image_np(one_face, face_shape)

      # DEBUG ONLY
      # extracted_face = cv2.cvtColor(dims_fixed, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(os.path.abspath(f'outputs_Combined/Night_Demo_Pat_data.mp4/frame_{random.randint(1, 1000)}.jpg'), extracted_face)

      #tf.image.resize_with_crop_or_pad(one_face, face_shape[0], face_shape[1])
      #torch_one = torch.tensor(dims_fixed)
      #faces[i] = torch_one.float()
      faces[i] = dims_fixed
    else:
      is_null[i] = 1
  #faces = torch.swapaxes(faces, 1, 3) / 255
  return faces, is_null

def detect_extract_one_face(one_im):
    face_shape=(224, 224, 3) 
    is_null = torch.zeros(1)
    faces = torch.empty((1, face_shape[0], face_shape[1], face_shape[2]), dtype=torch.float32)
    one_face = RetinaFace.extract_faces(one_im)
    if len(one_face) > 0:
        one_face = np.array(one_face[0])
        dims_fixed = letterbox_image_np(one_face, face_shape)
        #tf.image.resize_with_crop_or_pad(one_face, face_shape[0], face_shape[1])
        torch_one = torch.tensor(dims_fixed.numpy())
        faces[0] = torch_one.float()
    else:
        faces[0] = torch.zeros(face_shape[0], face_shape[1], face_shape[2]).float()
        is_null[0] = 1
    faces = torch.swapaxes(faces, 1, 3) / 255
    return faces, is_null 
   

import multiprocessing
import concurrent.futures

def multi_extract_faces(ims, face_shape=(224, 224, 3)):
   
    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Define the items to be processed
    items = ims

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Map the items to the worker processes in parallel
        results = list(executor.map(process_item, items))

    # Map the items to the worker processes in parallel
    results = pool.map(detect_extract_one_face, items)
    faces = [i for (i, j) in results]
    is_null = [j for (i, j) in results]


    return faces, is_null


"""

# ME-GraphAU & OpenGraphAU

"""

from megraphau.model.MEFL import MEFARG
from megraphau.OpenGraphAU.model.ANFL import MEFARG as MEFARG_OpenGraphAU
from collections import OrderedDict

class image_eval(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            # transforms.Resize(self.img_size),
            # transforms.CenterCrop(self.crop_size),
            # transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def load_network(model_type, backbone, path):
  if model_type == 'BP4D':
    net = MEFARG(num_classes=12, backbone=backbone)
    #path = 'megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth'
    net.load_state_dict(torch.load(path, map_location=torch.device(device)).get('state_dict'))
    
  elif model_type == 'OpenGraphAU':
    net = MEFARG_OpenGraphAU(num_main_classes=27, num_sub_classes=14, backbone=backbone, neighbor_num=4, metric='dots')
    
    #path = 'megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth'
    #path = 'megraphau/checkpoints/OpenGprahAU-ResNet50_second_stage.pth'
    oau_state_dict = torch.load(path, map_location=torch.device(device)).get('state_dict')
    
    oau_keys = oau_state_dict.keys()

    # filter out module.
    oau_state_dict_mod = OrderedDict([(k[7:], oau_state_dict[k]) if "module." in k else (k, oau_state_dict[k]) for k in oau_keys])
    
    net.load_state_dict(oau_state_dict_mod)

  net.eval()
  return net


def get_model_preds(faces, net, model_type):
  if use_cuda:
      print('Using GPU for Model Inference!')
      faces = faces.to(device)
      net = net.to(device)
  
  # If empty, return empty
  if faces.shape[0] == 0:
      return np.array([])

  with torch.no_grad():
    if model_type == 'BP4D':
      pred_ff = net(faces / 255)
      real_preds_ff = pred_ff[0].cpu().numpy()
    elif model_type == 'OpenGraphAU':
      pred_ff = net(faces)
      real_preds_ff = pred_ff.cpu().numpy()
  return real_preds_ff



"""

# Post-Processing

### Goal: increase stability of AU predictions, given the outputs (num frames  X  num AUs) of the network

"""

from pykalman import KalmanFilter

def simple_moving_average(X, window_size):
    """Calculate the simple moving average over a window of size window_size."""
    X_avg = np.zeros_like(X)
    for i in range(X.shape[0]):
        if i < window_size:
            X_avg[i] = np.mean(X[:i+1], axis=0)
        else:
            X_avg[i] = np.mean(X[i-window_size+1:i+1], axis=0)
    return X_avg


def exponential_moving_average(X, alpha):
    """Calculate the exponential moving average with decay parameter alpha."""
    X_ema = np.zeros_like(X)
    X_ema[0] = X[0]
    for i in range(1, X.shape[0]):
        X_ema[i] = alpha * X[i] + (1 - alpha) * X_ema[i-1]
    return X_ema



def kalman_filter(X):
    """Smooth the predictions using a Kalman filter."""
    kf = KalmanFilter(n_dim_obs=X.shape[1], n_dim_state=X.shape[1])
    X_kf, _ = kf.smooth(X)
    return X_kf



from scipy import signal

def temporal_difference(X):
    """Smooth the predictions using temporal difference."""
    # Compute the temporal differences
    diff = np.diff(X, axis=0)
    
    # Define the filter coefficients
    b = np.ones(5) / 5
    a = 1
    
    # Apply the filter to the differences
    diff_smoothed = signal.lfilter(b, a, diff, axis=0)
    
    # Integrate the smoothed differences
    X_smoothed = np.cumsum(np.vstack((X[0], diff_smoothed)), axis=0)
    return X_smoothed


def rnn_smoothing(X):
    """Smooth the predictions using a recurrent neural network."""
    # Define the RNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
        tf.keras.layers.Reshape((1, X.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(X.shape[1])
    ])
    
    # Compile the model
    model.compile(loss='mse', optimizer='adam')
    
    # Train the model
    model.fit(X, X, epochs=10, verbose=0)
    
    # Use the model to predict the smoothed output
    X_smoothed = model.predict(X)
    return X_smoothed

# from tslearn.metrics import dtw
# from tslearn.piecewise import PiecewiseAggregateApproximation

# def dtw_smoothing(X):
#     """Smooth the predictions using dynamic time warping."""
#     # Compute the piecewise aggregate approximation of the input data
#     n_segments = int(np.sqrt(X.shape[0]))
#     paa = PiecewiseAggregateApproximation(n_segments=n_segments)
#     X_paa = paa.fit_transform(X)
    
#     # Compute the distance matrix between all pairs of frames
#     distance_matrix = np.zeros((X.shape[0], X.shape[0]))
#     for i in range(X.shape[0]):
#         for j in range(X.shape[0]):
#             distance_matrix[i, j] = dtw(X_paa[i], X_paa[j])
    
#     # Compute the DTW-based similarity matrix
#     similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * np.median(distance_matrix) ** 2))
    
#     # Smooth the output using the similarity matrix
#     X_smoothed = np.zeros_like(X)
#     for i in range(X.shape[0]):
#         X_smoothed[i] = np.dot(similarity_matrix[i], X) / np.sum(similarity_matrix[i])
#     return X_smoothed


def postprocess_outs(preds, method='RNN'):
  method_dictionary = {
      'RNN': rnn_smoothing,
      #'DTW': dtw_smoothing,
      'TD': temporal_difference,
      'KF': kalman_filter,
      'EMA': lambda x: exponential_moving_average(x, EMA_ALPHA),
      'SMA': lambda x: simple_moving_average(x, SMA_WINDOW_SIZE)
  }

  postproc_func = method_dictionary.get(method)
  return postproc_func(preds)


"""

# CSV Saving

"""

def csv_save(labels, is_null, frames, save_path, fps):
    if labels.shape[0] == 0: # 0 frames successfully found in whole batch (due to downsampling)!
        labels = np.zeros((frames.shape[0], 41)) # We are defaulting to OpenGraphAU!
    
    elif labels.shape[1] == 12: # BP4D
        AU_ids = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
    elif labels.shape[1] == 41: # OpenGraphAU
        AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', 
                  '16', '17', '18', '19', '20', '22', '23', '24', '25', '26', '27', '32', 
                  '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 
                  'L12', 'R12', 'L14', 'R14']
    else: 
        print(f'Unexpected shape of labels! {labels.shape}') 
        return
    if not(os.path.exists(save_path)): # Make a new file with the correct first rows
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            row1 = ['frame', 'timestamp', 'success'] + ['AU' + i for i in AU_ids]
            writer.writerow(row1)
    
    # Make a modified array with (frame, timestamp, success) before AUs
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

def get_fps(path, extracting_fps=5):
  # image fps 
  capture = cv2.VideoCapture(path)
  fps = math.ceil(capture.get(cv2.CAP_PROP_FPS))

  # If video is corrupted, it will have 0 or very low fps, so actual fps is 0
  if fps < extracting_fps:
      return 0
  
  # we want extracting_fps fps. What did we get?
  frame_division = fps // extracting_fps
  actual_fps = fps / frame_division

  return actual_fps

def get_true_video_fps(path):
  # image fps 
  capture = cv2.VideoCapture(path)
  fps = math.ceil(capture.get(cv2.CAP_PROP_FPS))
  return fps



"""

# Presentation

"""


def draw_text_BP4D(img, probs):
    # BP4D labels
    AU_names = ['Inner brow raiser',
          'Outer brow raiser',
          'Brow lowerer',
          'Cheek raiser',
          'Lid tightener',
          'Upper lip raiser',
          'Lip corner puller',
          'Dimpler',
          'Lip corner depressor',
          'Chin raiser',
          'Lip tightener',
          'Lip pressor']
    AU_ids = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
    
    # from PIL import Image, ImageDraw, ImageFont
    #img = cv2.imread(path)
    pos_y = img.shape[0] // 40 + 20
    pos_x  = img.shape[1] + img.shape[1] // 100
    pos_x_ =  img.shape[1]  * 3 // 2 - img.shape[1] // 100

    img = cv2.copyMakeBorder(img, 0,0,0,img.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    # num_aus = len(words)
    # for i, item in enumerate(words):
    #     y = pos_y + (i * img.shape[0] // 17 )
    #     img = cv2.putText(img, str(item), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2048, 3), (0,0,255), 2)
    # pos_y = pos_y + (num_aus * img.shape[0] // 17 )
    #for i, item in enumerate(range(12)):
    for i, item in enumerate(range(len(AU_ids))):
        # y = pos_y  + (i * img.shape[0] // 13)
        y = pos_y  + (i * img.shape[0] // len(AU_ids) + 1)
        color = (0,0,0)
        if float(probs[item]) > 0.5:
            color = (0,0,255)
        img = cv2.putText(img,  AU_names[i] + ' -- AU' +AU_ids[i] +': {:.2f}'.format(probs[i]), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 1600, 3), color, 2)
        
    return img

def draw_text(img, probs):
    # OpenGraphAU
    AU_names = ['Inner brow raiser',
        'Outer brow raiser',
        'Brow lowerer',
        'Upper lid raiser',
        'Cheek raiser',
        'Lid tightener',
        'Nose wrinkler',
        'Upper lip raiser',
        'Nasolabial deepener',
        'Lip corner puller',
        'Sharp lip puller',
        'Dimpler',
        'Lip corner depressor',
        'Lower lip depressor',
        'Chin raiser',
        'Lip pucker',
        'Tongue show',
        'Lip stretcher',
        'Lip funneler',
        'Lip tightener',
        'Lip pressor',
        'Lips part',
        'Jaw drop',
        'Mouth stretch',
        'Lip bite',
        'Nostril dilator',
        'Nostril compressor',
        'Left Inner brow raiser',
        'Right Inner brow raiser',
        'Left Outer brow raiser',
        'Right Outer brow raiser',
        'Left Brow lowerer',
        'Right Brow lowerer',
        'Left Cheek raiser',
        'Right Cheek raiser',
        'Left Upper lip raiser',
        'Right Upper lip raiser',
        'Left Nasolabial deepener',
        'Right Nasolabial deepener',
        'Left Dimpler',
        'Right Dimpler']
    AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
           '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
    # from PIL import Image, ImageDraw, ImageFont
    pos_y = img.shape[0] // 40 + 15
    pos_x  = img.shape[1] + img.shape[1] // 100
    pos_x_ =  img.shape[1]  * 3 // 2 - img.shape[1] // 100

    img = cv2.copyMakeBorder(img, 0,0,0,img.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    # num_aus = len(words)
    # for i, item in enumerate(words):
    #     y = pos_y + (i * img.shape[0] // 17 )
    #     img = cv2.putText(img, str(item), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2048, 3), (0,0,255), 2)
    # pos_y = pos_y + (num_aus * img.shape[0] // 17 )
    for i, item in enumerate(range(21)):
        y = pos_y  + (i * img.shape[0] // 22)
        color = (0,0,0)
        if float(probs[item]) > 0.5:
            color = (0,0,255)
        img = cv2.putText(img,  AU_names[i] + ' -- AU' +AU_ids[i] +': {:.2f}'.format(probs[i]), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2800, 3), color, 2)

    for i, item in enumerate(range(21,41)):
        y = pos_y  + (i * img.shape[0] // 22)
        color = (0,0,0)
        if float(probs[item]) > 0.5:
            color = (0,0,255)
        img = cv2.putText(img,  AU_names[item] + ' -- AU' +AU_ids[item] +': {:.2f}'.format(probs[item]), (pos_x_, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2800, 3), color, 2)
    return img

def draw_text_openface(img, probs):
    # OpenFace
    AU_names = ['Inner brow raiser',
        'Outer brow raiser',
        'Brow lowerer',
        'Upper lid raiser',
        'Cheek raiser',
        'Lid tightener',
        'Nose wrinkler', #9
        'Upper lip raiser',
        'Lip corner puller',
        'Dimpler',
        'Lip corner depressor', #15
        'Chin raiser',
        'Lip stretcher', #20
        'Lip tightener', #23
        'Lips part',
        'Jaw drop', #26
        'Lip suck',
        'Blink']
    AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '28', '45']
    
    # from PIL import Image, ImageDraw, ImageFont
    pos_y = img.shape[0] // 40 + 15
    pos_x  = img.shape[1] + img.shape[1] // 100
    pos_x_ =  img.shape[1]  * 3 // 2 - img.shape[1] // 100

    img = cv2.copyMakeBorder(img, 0,0,0,img.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    # num_aus = len(words)
    # for i, item in enumerate(words):
    #     y = pos_y + (i * img.shape[0] // 17 )
    #     img = cv2.putText(img, str(item), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 2048, 3), (0,0,255), 2)
    # pos_y = pos_y + (num_aus * img.shape[0] // 17 )
    for i, item in enumerate(range(len(AU_ids))):
        # y = pos_y  + (i * img.shape[0] // 13)
        y = pos_y  + (i * img.shape[0] // len(AU_ids) + 1)
        color = (0,0,0)
        if float(probs[item]) > 0.5:
            color = (0,0,255)
        img = cv2.putText(img,  AU_names[i] + ' -- AU' +AU_ids[i] +': {:.2f}'.format(probs[i]), (pos_x, y), cv2.FONT_HERSHEY_SIMPLEX, round(img.shape[1] / 1600, 3), color, 2)
        
    return img


def extract_labels(images, preds, model_type):
  labeled_frames_ff = []
  for i in range(preds.shape[0]):
    if model_type == 'BP4D':
      labeled_frames_ff.append(draw_text_BP4D(images[i], preds[i]))
    elif model_type == 'OpenGraphAU':
      labeled_frames_ff.append(draw_text(images[i], preds[i]))
    elif model_type == 'OpenFace':
      labeled_frames_ff.append(draw_text_openface(images[i], preds[i]))
  labeled_frames_ff = np.array(labeled_frames_ff)
  return labeled_frames_ff


def save_video_from_images(images, video_name, fps=30):
    """
    Saves a video file from multiple images.

    Args:
    - images: a NumPy array of images
    - video_name: the name of the output video file
    - fps: the frame rate of the output video file (default is 30 fps)

    Returns:
    None
    """

    # Get the shape of the first image in the array
    height, width, channels = images[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Iterate through the images and write them to the video file
    for image in images:
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()


"""

# VIDEO PROCESSING

"""


def extract_specific_frames(input_video_path, frame_numbers, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number_set = set(frame_numbers)  # Convert to set for O(1) look-up time

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_number_set:
            out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

