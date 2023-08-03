"""

DEEPFACE

"""

import os
import pandas as pd
from deepface import DeepFace
import cv2
import shutil

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
        result = DeepFace.verify(img1_path=target_img_path, img2_path=data_now, enforce_detection=False, model_name='VGG-Face', detector_backend='mtcnn')

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



