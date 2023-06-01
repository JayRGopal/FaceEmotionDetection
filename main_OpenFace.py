import os
import subprocess
import time
import datetime
"""

Full Pipeline - OpenFace

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('inputs/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_OpenFace/')

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
    cmd = f'C:\\Users\\DannyHuang\\Desktop\\OpenFace_2.2.0_win_x64\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe -f "{video_path}" -out_dir "{OUTPUT_DIRECTORY}" -aus' 
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

  

  

