from utils import *
from utilsHSE import *
from utilsMMPose import *
import os
import time
import datetime
import cv2
import torch

# Device
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
if use_cuda:
  torch.cuda.empty_cache()

"""

Full Pipeline - HSE and OpenGraphAU
Detection via MTCNN
Verification using DeepFace (Model: VGG-Face)

"""

# Choose which pipelines to run
Run_HSE = True
Run_OpenGraphAU = True
Do_Verification = True # Toggling this isn't supported yet. Verification will always happen

# Set the parameters
BATCH_SIZE = 30
HSE_MODEL_TYPE = 'mobilenet_7.h5'
OPENGRAPHAU_MODEL_TYPE = 'OpenGraphAU'
OPENGRAPHAU_MODEL_BACKBONE = 'swin_transformer_base'
OPENGRAPHAU_MODEL_PATH = os.path.abspath('megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth')
INPUT_SIZE = (224, 224)
VIDEO_DIRECTORY = os.path.abspath('inputs/')
FPS_EXTRACTING = 5 # we'll extract 5 fps
OUTPUT_DIRECTORY = os.path.abspath('outputs_Combined') 
SUBJECT_FACE_IMAGE_PATH = os.path.abspath('deepface/Smiling_Face.jpg')  

# Function that gets us the output folder for each input video
SAVE_PATH_FOLDER = lambda video_name: os.path.join(OUTPUT_DIRECTORY, f'{video_name}')

# List of unprocessed videos
unprocessed_videos = get_valid_vids(VIDEO_DIRECTORY, SAVE_PATH_FOLDER)

# MMPose
TOP_DOWN = True
OUTPUT_VIDEO_DIRECTORY = OUTPUT_DIRECTORY # This is where videos/images with overlay go 
CONFIGS_BASE = os.path.abspath('mmpose/configs/body_2d_keypoint')
WHOLEBODY_CONFIGS_BASE = os.path.abspath('mmpose/configs/wholebody_2d_keypoint') 
MMPOSE_MODEL_BASE = os.path.abspath('MMPose_models/')
detector_mapping = get_detector_mapping(MMPOSE_MODEL_BASE)

# For timing estimation
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?

if Run_HSE:
  model_hse = get_emotion_predictor(HSE_MODEL_TYPE)

if Run_OpenGraphAU:
  model_ogau = load_network(model_type=OPENGRAPHAU_MODEL_TYPE, backbone=OPENGRAPHAU_MODEL_BACKBONE, path=OPENGRAPHAU_MODEL_PATH)


# Loop through all videos
for i in unprocessed_videos:
  video_path = os.path.join(VIDEO_DIRECTORY, i)

  frame_now = 0 # this is what we save in outputs file and print

  fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING) # FPS at which we're extracting

  save_folder_now = SAVE_PATH_FOLDER(i)

  
  os.mkdir(save_folder_now)
  save_path_hse = os.path.join(save_folder_now, f'outputs_hse.csv')
  save_path_ogau = os.path.join(save_folder_now, f'outputs_ogau.csv') 

  if TIMING_VERBOSE: 
    time1 = time.time()

  # Extract video frames
  capture = cv2.VideoCapture(video_path)
  ims = []
  real_frame_numbers = []
  real_fps = math.ceil(capture.get(cv2.CAP_PROP_FPS)) # real FPS of the video
  frame_division = real_fps // FPS_EXTRACTING # Helps us only analyze 5 fps (or close to it)
  running = True
  frameNr = 0 # Track frame number
  while running:
      # Extract frames continuously
      success, frame = capture.read()
      if success:
          if frameNr % frame_division == 0:
              # We are only saving SOME frames (e.g. extracting 5 fps)
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              ims.append(frame)
              real_frame_numbers.append(frameNr)
          if (frameNr % BATCH_SIZE == 0) and (frameNr > 0) and len(real_frame_numbers) > 0:
              # Let's do analysis, save results, and reset ims!
              ims = np.array(ims)
              print(f"Extracted Ims, Frames {frame_now} to {frameNr} in {i}") 
              if TIMING_VERBOSE:
                time2 = time.time()
                print('Time: ', time2 - time1)

              # Face detection
              faces, is_null = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_PATH)
              print(f"Detected Faces")
              if TIMING_VERBOSE:
                time3 = time.time()
                print('Time: ', time3 - time2) 

              # Get predictions of relevant network
              if Run_HSE:
                faces_for_hse = convert_to_gpu_tensor(faces)
                hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE)
                hse_scores_real[is_null == 1] = 0 # clear the predictions from frames w/o faces!
                print("Got Network Predictions: HSE")
              
              if use_cuda:
                torch.cuda.empty_cache()

              if Run_OpenGraphAU:
                image_evaluator = image_eval()
                faces_ogau = mtcnn_to_torch(faces)
                faces_ogau = image_evaluator(faces_ogau)
                faces_ogau = faces_ogau.to(device)
                ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE)
                ogau_predictions[is_null == 1] = 0 # clear the predictions from frames w/o faces!
                print("Got Network Predictions: OGAU")

              if use_cuda:
                torch.cuda.empty_cache()
              
              if TIMING_VERBOSE:
                time4 = time.time()
                print('Time: ', time4 - time3)

              # Save outputs to a CSV
              frames = np.array(real_frame_numbers).reshape(-1, 1)
              real_frame_numbers = []

              if Run_HSE:
                csv_save_HSE(labels=hse_scores_real, is_null=is_null, frames=frames, save_path=save_path_hse, fps=real_fps)
                print(f"Saved HSE CSV to {save_path_hse}!")
              
              if Run_OpenGraphAU:
                csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
                print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

              frame_now = frameNr

              # Reset ims for the next batch!
              ims = []

              # Reset timing
              if TIMING_VERBOSE: 
                time1 = time.time()
      else:
          # We're out of frames!
          running = False

          # Let's do analysis, save results, and reset ims!
          ims = np.array(ims)
          print(f"Extracted Ims, Frames {frame_now} to {frameNr} in {i}") 
          if TIMING_VERBOSE:
            time2 = time.time()
            print('Time: ', time2 - time1)

          # Face detection
          faces, is_null = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_PATH)
          print(f"Detected Faces")
          if TIMING_VERBOSE:
            time3 = time.time()
            print('Time: ', time3 - time2) 

          # Get predictions of relevant network
          if Run_HSE:
            faces_for_hse = convert_to_gpu_tensor(faces)
            hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE)    
            hse_scores_real[is_null == 1] = 0 # clear the predictions from frames w/o faces!
            print("Got Network Predictions: HSE")

          if use_cuda:
            torch.cuda.empty_cache()
          
          if Run_OpenGraphAU:
            image_evaluator = image_eval()
            faces_ogau = mtcnn_to_torch(faces)
            faces_ogau = image_evaluator(faces_ogau)
            faces_ogau = faces_ogau.to(device)
            ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE)
            ogau_predictions[is_null == 1] = 0 # clear the predictions from frames w/o faces!
            print("Got Network Predictions: OGAU")

          if use_cuda:
            torch.cuda.empty_cache()

          if TIMING_VERBOSE:
            time4 = time.time()
            print('Time: ', time4 - time3)

          # Save outputs to a CSV
          frames = np.array(real_frame_numbers).reshape(-1, 1)
          real_frame_numbers = []

          if Run_HSE:
            csv_save_HSE(labels=hse_scores_real, is_null=is_null, frames=frames, save_path=save_path_hse, fps=real_fps)
            print(f"Saved HSE CSV to {save_path_hse}!")

          if Run_OpenGraphAU:
            csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
            print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

          frame_now = frameNr

          # Reset ims to save space
          ims = []

          # Reset timing
          if TIMING_VERBOSE: 
            time1 = time.time()
      frameNr = frameNr + 1
  capture.release()

  # Time estimation
  elapsed_time = time.time() - start_time
  iterations_left = num_vids - unprocessed_videos.index(i) - 1
  time_per_iteration = elapsed_time / (unprocessed_videos.index(i) + 1)
  time_left = time_per_iteration * iterations_left
  time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
  
  # print an update on the progress
  print("Approximately ", time_left_formatted, " left to complete analyzing all videos")

 

