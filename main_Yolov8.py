from utils import *
from utilsHSE import *
from utilsVerify import *
import os
import time
import datetime
import cv2
import torch
import shutil
import subprocess

# Device
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
  torch.cuda.empty_cache()


"""

Full Pipeline - Yolov8 with Tracking
Verification using DeepFace (Model: VGG-Face)

Success column:
1: Successful!
0: Failure: Zero poses
-1: Failure: Pose detected, and verification failed (across all images in loop!)

"""

# Choose which pipelines to run
Do_Verification = True 

# Set the parameters
BATCH_SIZE = 2000
INPUT_SIZE = (224, 224)
VIDEO_DIRECTORY = os.path.abspath('inputs/')
FPS_EXTRACTING = 5 # we'll extract this many fps from the video for analysis
OUTPUT_DIRECTORY = os.path.abspath('outputs_Yolov8') 
SUBJECT_FACE_IMAGE_FOLDER = os.path.abspath('deepface/')

# Function that gets us the output folder for each input video
SAVE_PATH_FOLDER = lambda video_name: os.path.join(OUTPUT_DIRECTORY, f'{video_name}')

# List of unprocessed videos
unprocessed_videos = get_valid_vids(VIDEO_DIRECTORY, SAVE_PATH_FOLDER)

# For timing estimation
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?


# Loop through all videos
for i in unprocessed_videos:
  video_path = os.path.join(VIDEO_DIRECTORY, i)

  frame_now = 0 # this is what we save in outputs file and print

  fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING) # FPS at which we're extracting

  # Save paths/folders
  save_folder_now = SAVE_PATH_FOLDER(i)
  os.mkdir(save_folder_now)
  save_path_yolov8 = os.path.join(save_folder_now, f'outputs_yolov8.csv')

  # Clear results
  yolov8_results_folder = 'yolo_tracking/runs/track'
  if os.path.exists(yolov8_results_folder):
    shutil.rmtree(yolov8_results_folder)

  os.chdir('yolo_tracking')
  cmd = f'python3 examples/track.py --yolo-model yolov8n-pose.pt --reid-model clip_market1501.pt --tracking-method botsort \
  --source {video_path} --conf 0.2 --classes 0 --save-txt --vid-stride {FPS_EXTRACTING} --save-mot --show-labels' 
  subprocess.run(cmd, shell=True)
  os.chdir('..')
  yolov8_this_run_outputs = 'yolo_tracking/runs/track/exp'


  if Do_Verification:
    # Extract video frames for verification
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

                
        else:
            # We're out of frames!
            running = False

            # Let's do analysis, save results, and reset ims!
            ims = np.array(ims)
            print(f"Extracted Ims, Frames {frame_now} to {frameNr} in {i}")

            
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

 

