import os
import subprocess
import time
import datetime
from setup import download_file
import torch

from utilsMMPose import *

"""

Full Pipeline - MMPose

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('testing_images/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_MMPose/')
TOP_DOWN = True
DOWNLOAD_DIRECTORY = os.path.abspath('MMPose_models/')
CONFIGS_BASE = os.path.abspath('mmpose/configs/body_2d_keypoint') 


# Model setup list
# (config_path, model_download, model_path)
model_setup_list = [
  (f'{CONFIGS_BASE}/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py', 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth', 'MMPose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
]

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# For timing estimation
valid_videos = [vid for vid in all_videos if os.path.isfile(os.path.join(VIDEO_DIRECTORY, vid))]
num_vids = len(valid_videos)
start_time = time.time()

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loop through all model setups
for (config_file, model_download, model_path) in model_setup_list:

  # Download model if not already there
  if not(os.path.exists(model_path)): 
    download_file(model_download, model_path)
  
  model_base = os.path.split(model_path)[-1]
  os.makedirs(os.path.join(OUTPUT_DIRECTORY, model_base), exist_ok=True)

  df_list = []

  # Loop through all videos
  for i in all_videos:
    save_file = os.path.join(OUTPUT_DIRECTORY, f'{model_base}', 'results_' + i[:-4] + '.json') 
    video_path = os.path.join(VIDEO_DIRECTORY, i)
    if os.path.exists(save_file):
      print(f'Skipping Video {i}: Output File Already Exists!')
    elif os.path.isfile(video_path):
      if TOP_DOWN:
        
        cmd = f'python mmpose/JayGopal/run_topdown.py mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
          MMPose_models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
          "{os.path.abspath(config_file)}" \
          "{os.path.abspath(model_path)}" \
          --input "{video_path}" \
          --draw-heatmap \
          --save-predictions \
          --output-root "{os.path.abspath(f"{OUTPUT_DIRECTORY}/{model_base}/")}" \
          --device {device}' 
      else:
        cmd = f'python mmpose/JayGopal/run_bottomup.py \
          "{os.path.abspath(config_file)}" \
          "{os.path.abspath(model_path)}" \
          --input "{video_path}" \
          --output-root "{os.path.abspath(f"{OUTPUT_DIRECTORY}/{model_base}/")}" \
          --save-predictions --draw-heatmap \
          --device {device}'

      subprocess.run(cmd, shell=True)
      df_list.append(convert_to_df(save_file)) 
      
    else:
      print(f'WARNING: Got path {video_path}, which is not a valid video file!')
  df_combined = pd.concat(df_list, ignore_index=True)
  df_combined.insert(0, 'Filename', valid_videos)
  df_combined.to_csv(os.path.join(OUTPUT_DIRECTORY, f'{model_base}/combined.csv'), index=False)
  # Time estimation
  elapsed_time = time.time() - start_time
  iterations_left = len(model_setup_list) - model_setup_list.index( (config_file, model_download, model_path) ) - 1
  time_per_iteration = elapsed_time / (model_setup_list.index( (config_file, model_download, model_path) ) + 1)
  time_left = time_per_iteration * iterations_left
  time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
  
  # print an update on the progress
  print("Approximately", time_left_formatted, "left to complete the operation")
  



  

  

