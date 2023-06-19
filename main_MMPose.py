import os
import subprocess
import time
import datetime
"""

Full Pipeline - MMPose

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('inputs/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_MMPose/')
TOP_DOWN = True
DOWNLOAD_DIRECTORY = os.path.abspath('MMPose_models/')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# For timing estimation
valid_videos = [vid for vid in all_videos if os.path.isfile(os.path.join(VIDEO_DIRECTORY, vid))]
unprocessed_videos = [vid for vid in valid_videos if not(os.path.exists(os.path.join(OUTPUT_DIRECTORY, vid[:-4] + '.csv') ))]
num_vids = len(unprocessed_videos)
start_time = time.time()

# Loop through all videos
for i in all_videos:
  save_file = os.path.join(OUTPUT_DIRECTORY, i[:-4] + '.csv') 
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  if os.path.exists(save_file):
    print(f'Skipping Video {i}: Output File Already Exists!')
  elif os.path.isfile(video_path):
    if TOP_DOWN:
      cmd = f'python mmpose/JayGopal/run_topdown.py \
        demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
        configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
        https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
        --input tests/data/coco/000000197388.jpg --draw-heatmap \
        --save-predictions \
        --output-root vis_results_topdown/' 
    else:

    subprocess.run(cmd, shell=True)
     
    # Time estimation
    elapsed_time = time.time() - start_time
    iterations_left = num_vids - unprocessed_videos.index(i) - 1
    time_per_iteration = elapsed_time / (unprocessed_videos.index(i) + 1)
    time_left = time_per_iteration * iterations_left
    time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
    
    # print an update on the progress
    print("Approximately", time_left_formatted, "left to complete the operation")
  else:
    print(f'WARNING: Got path {video_path}, which is not a valid video file!')

  

  

