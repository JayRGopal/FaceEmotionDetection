"""

DEEPFACE

"""

import os
import pandas as pd
from deepface import DeepFace
import cv2
import shutil
import sys
import numpy as np

def verify_faces_np_data(target_img_path, np_data):
    # Goal: determine which images have the target face, and get the bboxes of the target face in those images.
    # Returns a pandas df that has an 'index' column indicating index in np_data, and the bbox coordinates for each index
    # Note that our final pandas df won't have all indices in np_data since some frames won't have successful verification of our target face!

    # Verifying each image
    results = []
    for i in range(np_data.shape[0]):
        data_now = np_data[i]
        # Undo preprocessing
        data_now = cv2.cvtColor(data_now, cv2.COLOR_RGB2BGR) 

        # SILENT RUN
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        result = DeepFace.verify(img1_path=target_img_path, img2_path=data_now, enforce_detection=False, model_name='VGG-Face', detector_backend='mtcnn')
        sys.stdout.close()
        sys.stdout = original_stdout

        if result['verified']:
            face_x = result['facial_areas']['img2']['x']
            face_y = result['facial_areas']['img2']['y']
            face_w = result['facial_areas']['img2']['w']
            face_h = result['facial_areas']['img2']['h']
            image_data = {
                'Index': int(i),
                'Distance': result['distance'],
                'Facial Box X': int(face_x),
                'Facial Box Y': int(face_y),
                'Facial Box W': int(face_w),
                'Facial Box H': int(face_h)
            }
            results.append(image_data)

    # Getting a pandas df
    df = pd.DataFrame(results)
    df.columns = ['Index', 'Distance', 'Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']
    
    return df

def verify_one_face_np_data(target_img_path, np_data):
    # Goal: determine if one image has the target face, and get the bboxes of the target face in that image.
    # Returns either a 4-membered tuple (x, y, w, h) or None depending on whether face was verified
    
    # Verifying each image
    
    data_now = np_data
    # Undo preprocessing
    data_now = cv2.cvtColor(data_now, cv2.COLOR_RGB2BGR) 

    # SILENT RUN
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    result = DeepFace.verify(img1_path=target_img_path, img2_path=data_now, enforce_detection=False, model_name='VGG-Face', detector_backend='mtcnn')
    sys.stdout.close()
    sys.stdout = original_stdout

    if result['verified']:
        face_x = result['facial_areas']['img2']['x']
        face_y = result['facial_areas']['img2']['y']
        face_w = result['facial_areas']['img2']['w']
        face_h = result['facial_areas']['img2']['h']
        return_tuple = (face_x, face_y, face_w, face_h)
        return return_tuple
    else:
        return None 



"""

MMPose

"""


def avg_face_conf_body_2d(pose_results):
  # Given a list of MMPose poseResult objects
  # Returns a (num_faces,) array with average face confidence for each face
  # Assumes there is at least 1 face detected (len(pose_results) >= 1)
  # NOTE: Assumes body 2d keypoints, which has a specific number of facial landmarks!

  if len(pose_results) < 1:
    raise ValueError("Pose Results must have at least 1 face detected!")
  
  confidences = []
  for i in pose_results:
    preds = i.get('pred_instances')
    kp_scores = preds['keypoint_scores']
    face_confidences = kp_scores[0][0:3] # nose and two eyes
    avg_conf = np.mean(face_confidences)
    confidences.append(avg_conf)
  confidences = np.array(confidences)
  return confidences 

def get_nose_coords_body_2d(pose_results):
  # Given a list of MMPose poseResult objects
  # Returns a (num_faces,2) array with nose x, nose y for each face
  # Assumes there is at least 1 face detected (len(pose_results) >= 1)
  # NOTE: Assumes body 2d keypoints, which has a specific number of facial landmarks!

  if len(pose_results) < 1:
    raise ValueError("Pose Results must have at least 1 face detected!")
  
  nose_coords_all = []
  for i in pose_results:
    preds = i.get('pred_instances')
    kp_coords = preds['keypoints']
    nose_coords = kp_coords[0][0]
    nose_coords_all.append(nose_coords)
  nose_coords_all = np.array(nose_coords_all)
  return nose_coords_all 

  


def closest_person_index(correct_x, correct_y, nose_coordinates, threshold=25, printMins=True):
    # Compute the Euclidean distances from (correct_x, correct_y) to all points in nose_coordinates
    distances = np.sqrt((nose_coordinates[:, 0] - correct_x)**2 + (nose_coordinates[:, 1] - correct_y)**2)
    
    # Get the index of the closest person
    min_index = np.argmin(distances)

    # Printing
    if printMins:
      print('MINIMUM DISTANCE FOR ONE FRAME:', distances[min_index])
    
    # Check if the closest distance is greater than the threshold
    if distances[min_index] > threshold:
        return -1
    else:
        return min_index

"""

FACETORCH (Not using)

"""

from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity

def load_config(path_to_config="./facetorch/facetorch_config_verifyOnly.yml"):
  return OmegaConf.load(path_to_config)

def init_analyzer(cfg, path_image='./facetorch/demo.jpg'):
  # initialize
  analyzer = FaceAnalyzer(cfg.analyzer)

  # warmup
  response = analyzer.run(
          path_image=path_image,
          batch_size=cfg.batch_size,
          fix_img_size=cfg.fix_img_size,
          return_img_data=False,
          include_tensors=True,
          path_output='./facetorch/im_out.jpg')
  
  return analyzer


def get_verify_vectors_one_face(analyzer, cfg, face_data, tmp_save='./facetorch/tmp_save.jpg'):
  # Save Image to file
  cv2.imwrite(os.path.join(tmp_save, f'frame_0.jpg'), face_data)

  # Get Facetorch output
  response = analyzer.run(
        path_image=tmp_save,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output='./facetorch/im_out.jpg')

  return response


