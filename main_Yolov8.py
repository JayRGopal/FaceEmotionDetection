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
Do_Verification = False # Verify not supported yet! 

# Set the parameters
BATCH_SIZE = 2000
INPUT_SIZE = (224, 224)
VIDEO_DIRECTORY = os.path.abspath('inputs/')
VID_STRIDE = 5 # we'll extract one frame per every VID_STRIDE frames in the video!
OUTPUT_DIRECTORY = os.path.abspath('outputs_Yolov8') 
OUTPUT_DIRECTORY_POSE_OVERLAYS = os.path.abspath('outputs_Yolov8_PatData') 
SUBJECT_FACE_IMAGE_FOLDER = os.path.abspath('deepface/')

# List of unprocessed videos
unprocessed_videos = get_valid_vids(VIDEO_DIRECTORY, lambda video_name: os.path.join(OUTPUT_DIRECTORY, f'{video_name}'))

# For timing estimation
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?


# Loop through all videos
for i in unprocessed_videos:
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  
  fps = get_true_video_fps(video_path) # FPS of video

  os.chdir('yolo_tracking')
  cmd = f'python3 examples/track.py --yolo-model yolov8n-pose.pt --reid-model clip_market1501.pt --tracking-method botsort \
  --source "{video_path}" --conf 0.2 --classes 0 --vid-stride {VID_STRIDE} --save --vid-fps {fps} --project "{OUTPUT_DIRECTORY}" --project-patdata "{OUTPUT_DIRECTORY_POSE_OVERLAYS}"' 
  subprocess.run(cmd, shell=True)
  os.chdir('..')

  # Time estimation
  elapsed_time = time.time() - start_time
  iterations_left = num_vids - unprocessed_videos.index(i) - 1
  time_per_iteration = elapsed_time / (unprocessed_videos.index(i) + 1)
  time_left = time_per_iteration * iterations_left
  time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
  
  # print an update on the progress
  print("Approximately ", time_left_formatted, " left to complete analyzing all videos")

 

