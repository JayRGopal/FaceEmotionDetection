from utils import *
from utilsHSE import *
from utilsVerify import *
import os
import time
import datetime
import cv2
import torch
import argparse
from facenet_pytorch import MTCNN

# Main parameters
VIDEO_DIRECTORY = os.path.abspath('inputs/')
SUBJECT_FACE_IMAGE_FOLDER = os.path.abspath('deepface/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_Combined_Outpt/Pt1/') 
OUTPUT_DIRECTORY_PARTIAL_VERIFY = os.path.abspath('outputs_Combined_OutPt_PatData/Pt1/') 


# Device
def create_parser():
    parser = argparse.ArgumentParser(description='Process device information.')

    # Add the 'device' argument
    # It accepts values like 'cuda:0', 'cpu', or 'cuda:1'
    parser.add_argument('--device', type=str, required=True,
                        help='Specify the device to use, e.g., cuda:0, cpu, cuda:1')
    
    # Forcing HSE to CPU?
    parser.add_argument('--force-hse-cpu', action='store_true',
                        help='Force the use of HSE CPU if set')

    return parser

parser = create_parser()
args = parser.parse_args()
device = args.device
FORCE_HSE_CPU = args.force_hse_cpu

if 'cuda' in device:
  use_cuda = True
else:
  use_cuda = False

if use_cuda:
  torch.cuda.empty_cache()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

"""

Full Pipeline - HSE and OpenGraphAU
Detection via MTCNN
Verification using DeepFace (Model: VGG-Face)

Success column:
1: Successful!
0: Failure: Zero faces detected
-1: Failure: 2+ faces detected, and verification failed (across all images in loop!)
-2: Failure: 1 face detected, and verification failed (across all images in loop). Only possible with Verify_Every_Frame = True

"""

# Choose which pipelines to run
Run_HSE = True
Run_OpenGraphAU = True
Do_Verification = True 
Partial_Verify = False # Verify, then find nearest face within distance max (below)
Verify_Every_Frame = True # Verify all frames, even if only 1 person is detected
VERIFY_THRESHOLD = 0.32 # Maximum distance threshold (below this, faces are deemed "verified")
Face_Detector = 'MTCNN' # Options: ['MTCNN', 'RetinaFace']

# Set the parameters
BATCH_SIZE = 2000
HSE_MODEL_TYPE = 'mobilenet_7.h5'
OPENGRAPHAU_MODEL_TYPE = 'OpenGraphAU'
OPENGRAPHAU_MODEL_BACKBONE = 'swin_transformer_base'
OPENGRAPHAU_MODEL_PATH = os.path.abspath('megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth')
INPUT_SIZE = (224, 224)
FPS_EXTRACTING = 5 # we'll extract this many fps from the video for analysis
DISTANCE_MAX_PARTIAL_VERIFY = 30
SAVE_PROB_PARTIAL_VERIFY = 0.01

# Make output dirs if they don't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists(OUTPUT_DIRECTORY_PARTIAL_VERIFY):
    os.makedirs(OUTPUT_DIRECTORY_PARTIAL_VERIFY)

# Function that gets us the output folder for each input video
SAVE_PATH_FOLDER = lambda video_name: os.path.join(OUTPUT_DIRECTORY, f'{video_name}')
SAVE_PATH_FOLDER_PARTIAL_VERIFY = lambda video_name: os.path.join(OUTPUT_DIRECTORY_PARTIAL_VERIFY, f'{video_name}')

# List of unprocessed videos
unprocessed_videos = get_valid_vids(VIDEO_DIRECTORY, SAVE_PATH_FOLDER)

# For timing estimation
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?

# Load models
mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)

if Run_HSE:
  model_hse = get_emotion_predictor(HSE_MODEL_TYPE, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)

if Run_OpenGraphAU:
  model_ogau = load_network(model_type=OPENGRAPHAU_MODEL_TYPE, backbone=OPENGRAPHAU_MODEL_BACKBONE, path=OPENGRAPHAU_MODEL_PATH, device=device)

if Do_Verification:
  assert has_jpg_or_jpeg_files(SUBJECT_FACE_IMAGE_FOLDER), "No jpg or jpeg files found in SUBJECT_FACE_IMAGE_FOLDER. Can't do facial verification!"


# Loop through all videos
for i in unprocessed_videos:
  video_path = os.path.join(VIDEO_DIRECTORY, i)

  frame_now = 0 # this is what we save in outputs file and print

  running = True
  
  fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING) # FPS at which we're extracting

  # If video is corrupted, skip it
  if fps == 0:
    running = False
    print('-'*15)
    print(f'MAJOR WARNING! SKIPPING CORRUPTED VIDEO: {i}')
    print('-'*15)

  # Save paths/folders
  save_folder_now = SAVE_PATH_FOLDER(i)
  save_folder_partial_verify_now = SAVE_PATH_FOLDER_PARTIAL_VERIFY(i)
  os.mkdir(save_folder_now)
  if Do_Verification and Partial_Verify:
    os.makedirs(save_folder_partial_verify_now, exist_ok=True)
  save_path_hse = os.path.join(save_folder_now, f'outputs_hse.csv')
  save_path_ogau = os.path.join(save_folder_now, f'outputs_ogau.csv') 
  save_path_bboxes = os.path.join(save_folder_now, f'outputs_bboxes.csv') 

  if TIMING_VERBOSE: 
    time1 = time.time()

  # Extract video frames
  capture = cv2.VideoCapture(video_path)
  ims = []
  real_frame_numbers = []
  real_fps = math.ceil(capture.get(cv2.CAP_PROP_FPS)) # real FPS of the video
  frame_division = real_fps // FPS_EXTRACTING # Helps us only analyze 5 fps (or close to it)
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
              if Do_Verification:
                if Face_Detector == 'MTCNN':
                  faces, is_null, all_bboxes = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_FOLDER, partialVerify=Partial_Verify, \
                                                                         mtcnn=mtcnn, verifyAll=Verify_Every_Frame, verify_threshold=VERIFY_THRESHOLD, \
                                                             distance_max=DISTANCE_MAX_PARTIAL_VERIFY, save_folder_path=save_folder_partial_verify_now, \
                                                              real_frame_numbers=real_frame_numbers, saveProb=SAVE_PROB_PARTIAL_VERIFY)
                elif Face_Detector == 'RetinaFace':
                  faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
              else:
                if Face_Detector == 'MTCNN':
                  faces, is_null, all_bboxes = extract_faces_mtcnn(ims, INPUT_SIZE, mtcnn=mtcnn, real_frame_numbers=real_frame_numbers)
                elif Face_Detector == 'RetinaFace':
                  faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
              print(f"Detected Faces")
              if TIMING_VERBOSE:
                time3 = time.time()
                print('Time: ', time3 - time2) 

              # Get predictions of relevant network
              if Run_HSE:
                faces_for_hse = convert_to_gpu_tensor(faces, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)
                hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE, device=device)
                hse_scores_real[is_null > 0] = 0 # clear the predictions from frames w/o faces!
                del faces_for_hse
                print("Got Network Predictions: HSE")
              
              if use_cuda:
                torch.cuda.empty_cache()

              if Run_OpenGraphAU:
                image_evaluator = image_eval()
                faces_ogau = mtcnn_to_torch(faces)
                faces_ogau = image_evaluator(faces_ogau)
                faces_ogau = faces_ogau.to(device)
                ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE, device=device)
                ogau_predictions[is_null > 0] = 0 # clear the predictions from frames w/o faces!
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
                del hse_scores_real
                print(f"Saved HSE CSV to {save_path_hse}!")
              
              if Run_OpenGraphAU:
                csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
                print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

              if 'all_bboxes' in globals():
                csv_save_bboxes(labels=all_bboxes[['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']].values,  is_null=is_null, frames=frames, save_path=save_path_bboxes, fps=real_fps)
                print(f"Saved Facial Bboxes CSV to {save_path_bboxes}!")
              
              frame_now = frameNr

              # Reset ims for the next batch!
              del ims
              del frames
              del faces
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
          if Do_Verification:
            if Face_Detector == 'MTCNN':
              faces, is_null, all_bboxes = extract_faces_with_verify(ims, INPUT_SIZE, SUBJECT_FACE_IMAGE_FOLDER, partialVerify=Partial_Verify, \
                                                                     mtcnn=mtcnn, verifyAll=Verify_Every_Frame, verify_threshold=VERIFY_THRESHOLD, \
                                                         distance_max=DISTANCE_MAX_PARTIAL_VERIFY, save_folder_path=save_folder_partial_verify_now, \
                                                          real_frame_numbers=real_frame_numbers, saveProb=SAVE_PROB_PARTIAL_VERIFY)
            elif Face_Detector == 'RetinaFace':
              faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
          else:
            if Face_Detector == 'MTCNN':
              faces, is_null, all_bboxes = extract_faces_mtcnn(ims, INPUT_SIZE, mtcnn=mtcnn, real_frame_numbers=real_frame_numbers)
            elif Face_Detector == 'RetinaFace':
              faces, is_null = detect_extract_faces(ims, INPUT_SIZE)
          print(f"Detected Faces")
          if TIMING_VERBOSE:
            time3 = time.time()
            print('Time: ', time3 - time2) 
          
          # Get predictions of relevant network
          if Run_HSE:
            faces_for_hse = convert_to_gpu_tensor(faces, device=device, FORCE_HSE_CPU=FORCE_HSE_CPU)
            hse_scores_real = hse_preds(faces_for_hse, model_hse, model_type=HSE_MODEL_TYPE, device=device)    
            hse_scores_real[is_null > 0] = 0 # clear the predictions from frames w/o faces!
            del faces_for_hse
            print("Got Network Predictions: HSE")

          if use_cuda:
            torch.cuda.empty_cache()
          
          if Run_OpenGraphAU:
            image_evaluator = image_eval()
            faces_ogau = mtcnn_to_torch(faces)
            faces_ogau = image_evaluator(faces_ogau)
            faces_ogau = faces_ogau.to(device)
            ogau_predictions = get_model_preds(faces_ogau, model_ogau, model_type=OPENGRAPHAU_MODEL_TYPE, device=device)
            ogau_predictions[is_null > 0] = 0 # clear the predictions from frames w/o faces!
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
            del hse_scores_real
            print(f"Saved HSE CSV to {save_path_hse}!")

          if Run_OpenGraphAU:
            csv_save(labels=ogau_predictions, is_null=is_null, frames=frames, save_path=save_path_ogau, fps=real_fps)
            print(f"Saved OpenGraphAU CSV to {save_path_ogau}!")

          if 'all_bboxes' in globals():
            csv_save_bboxes(labels=all_bboxes[['Facial Box X', 'Facial Box Y', 'Facial Box W', 'Facial Box H']].values,  is_null=is_null, frames=frames, save_path=save_path_bboxes, fps=real_fps)
            print(f"Saved Facial Bboxes CSV to {save_path_bboxes}!")

          frame_now = frameNr

          # Reset ims to save space
          del ims
          del frames
          del faces
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

 

