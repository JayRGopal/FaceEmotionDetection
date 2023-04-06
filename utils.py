import numpy as np
import tensorflow as tf
import torch
import cv2


"""

# Video to Image Frames

"""

def extract_images(path, start_frame, num_to_extract): 
  capture = cv2.VideoCapture(path)

  ims = []
  frameNr = 0

  while frameNr < (start_frame + num_to_extract):
      success, frame = capture.read()
      if success:
          if frameNr >= start_frame:
            ims.append(frame)
      else:
          break
      frameNr = frameNr+1
  capture.release()

  ims = np.array(ims)

  im_test = torch.tensor(np.resize(ims[0:ims.shape[0]], (ims.shape[0], 3, 244, 244))).type(torch.float32)
  return (ims, im_test)



"""

# Face Detection (RetinaFace)

"""

from retinaface import RetinaFace

def detect_extract_faces(ims, face_shape=(224, 224, 3)):
  NUM_TO_EXTRACT = ims.shape[0]
  faces = torch.empty((NUM_TO_EXTRACT, face_shape[0], face_shape[1], face_shape[2]), dtype=torch.float32)
  for i in range(NUM_TO_EXTRACT):
    one_face = RetinaFace.extract_faces(ims[i])
    if len(one_face) > 0:
      one_face = np.array(one_face[0])
      dims_fixed = tf.image.resize_with_crop_or_pad(one_face, face_shape[0], face_shape[1])
      torch_one = torch.tensor(dims_fixed.numpy())
      faces[i] = torch_one.float()
    else:
      faces[i] = torch.zeros(face_shape[0], face_shape[1], face_shape[2]).float()
  faces = torch.swapaxes(faces, 1, 3) / 255
  return faces



"""

# ME-GraphAU & OpenGraphAU

"""

from me-graphau.model.MEFL import MEFARG
from me-graphau.OpenGraphAU.model.ANFL import MEFARG as MEFARG_OpenGraphAU
from me-graphau.collections import OrderedDict


def load_network(model_type, backbone):
  if model_type == 'BP4D':
    net = MEFARG(num_classes=12, backbone=backbone)
    path = 'me-graphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth'
    net.load_state_dict(torch.load(path).get('state_dict'))
  elif model_type == 'OpenGraphAU':
    net = MEFARG_OpenGraphAU(num_main_classes=27, num_sub_classes=14, backbone=backbone, neighbor_num=4)

    path = 'me-graphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth'
    oau_state_dict = torch.load(path).get('state_dict')
    oau_keys = oau_state_dict.keys()

    # filter out module.
    oau_state_dict_mod = OrderedDict([(k[7:], oau_state_dict[k]) if "module." in k else (k, oau_state_dict[k]) for k in oau_keys])
    
    net.load_state_dict(oau_state_dict_mod)

  net.eval()
  return net


def get_model_preds(faces, net, model_type):
  if torch.cuda.is_available():
      faces = faces.cuda()
      net = net.cuda()

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

import tensorflow as tf

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

from tslearn.metrics import dtw
from tslearn.piecewise import PiecewiseAggregateApproximation

def dtw_smoothing(X):
    """Smooth the predictions using dynamic time warping."""
    # Compute the piecewise aggregate approximation of the input data
    n_segments = int(np.sqrt(X.shape[0]))
    paa = PiecewiseAggregateApproximation(n_segments=n_segments)
    X_paa = paa.fit_transform(X)
    
    # Compute the distance matrix between all pairs of frames
    distance_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distance_matrix[i, j] = dtw(X_paa[i], X_paa[j])
    
    # Compute the DTW-based similarity matrix
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * np.median(distance_matrix) ** 2))
    
    # Smooth the output using the similarity matrix
    X_smoothed = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_smoothed[i] = np.dot(similarity_matrix[i], X) / np.sum(similarity_matrix[i])
    return X_smoothed


def postprocess_outs(preds, method='RNN'):
  method_dictionary = {
      'RNN': rnn_smoothing,
      'DTW': dtw_smoothing,
      'TD': temporal_difference,
      'KF': kalman_filter,
      'EMA': exponential_moving_average,
      'SMA': simple_moving_average
  }

  postproc_func = method_dictionary.get(method)
  return postproc_func(preds)




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

