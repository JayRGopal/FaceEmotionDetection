import os
import subprocess
"""

Full Pipeline - OpenFace

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('inputs/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_OpenFace/')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# Loop through all videos
for i in all_videos:
  save_file = os.path.join(OUTPUT_DIRECTORY, i[:-4] + '.csv') 
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  if os.path.exists(save_file):
    print(f'Skipping Video {i}: Output File Already Exists!')
  elif os.path.isfile(video_path):
    cmd = f'C:\\Users\\DannyHuang\\Desktop\\OpenFace_2.2.0_win_x64\\OpenFace_2.2.0_win_x64\\FeatureExtraction.exe -f "{video_path}" -out_dir "{OUTPUT_DIRECTORY}" -aus' 
    subprocess.run(cmd, shell=True)
  else:
    print(f'WARNING: Got path {video_path}, which is not a valid video file!')

  

  

