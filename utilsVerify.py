"""

DEEPFACE

"""

import os
import pandas as pd
from deepface import DeepFace
import cv2
import shutil

def verify_images(target_img_path, folder_path):
    # returns a pandas df of all images in folder_path that have the face in target_img_path
    # each row will have the bbox of the verified face (X, Y, W, H)

    results = []

    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):
            img_path = os.path.join(folder_path, image_file)
            result = DeepFace.verify(img1_path=target_img_path, img2_path=img_path, enforce_detection=False, model_name='ArcFace')

            if result['verified']:
                face_x = result['facial_areas']['img2']['x']
                face_y = result['facial_areas']['img2']['y']
                face_w = result['facial_areas']['img2']['w']
                face_h = result['facial_areas']['img2']['h']
                image_data = {
                    'Image File': img_path,
                    'Distance': result['distance'],
                    'Facial Box X': face_x,
                    'Facial Box Y': face_y,
                    'Facial Box W': face_w,
                    'Facial Box H': face_h
                }
                results.append(image_data)

    results_df = pd.DataFrame(results)
    return results_df


def verify_faces_np_data(target_img_path, np_data, tmp_save=os.path.abspath('tmp_save_deepface/')):
    # Goal: determine which images have the target face, and get the bboxes of the target face in those images.
    # Returns a pandas df that has an 'index' column indicating index in np_data, and the bbox coordinates for each index
    # Note that our final pandas df won't have all indices in np_data since some frames won't have successful verification of our target face!
 

    # Creating tmp_save folder if doesn't exist
    os.makedirs(tmp_save, exist_ok=True)
    
    # Deleting everything in the tmp_save folder if anything exists
    for filename in os.listdir(tmp_save):
        file_path = os.path.join(tmp_save, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Saving each frame in np_data in sequential order to the tmp_save folder
    for i in range(np_data.shape[0]):
        cv2.imwrite(os.path.join(tmp_save, f'frame_{i}.jpg'), np_data[i])

    # Getting a pandas df by calling verify_images(target_img_path, tmp_save)
    df = verify_images(target_img_path, tmp_save)

    # Deleting everything in the tmp_save folder if anything exists
    for filename in os.listdir(tmp_save):
        file_path = os.path.join(tmp_save, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Adding index column indicating index in np_data
    df['index'] = df['Image File'].apply(lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    # Reordering the columns
    df = df[['index', 'Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']]

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

  # Get Facetorch output
  response = analyzer.run(
        path_image=tmp_save,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output='./facetorch/im_out.jpg')

return



