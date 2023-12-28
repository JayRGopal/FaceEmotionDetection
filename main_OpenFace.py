import os
import subprocess
import time
import datetime
"""


Full Pipeline - OpenFace


NOTE: Before running this, you need to have already installed OpenFace and put the binary at the appropriate path.
If you're on Windows, use this link: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation


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
  txt_path = os.path.join(OUTPUT_DIRECTORY, i[:-4] + '_of_details.txt')
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  if os.path.exists(save_file):
    print(f'Skipping Video {i}: Output File Already Exists!')
  elif os.path.isfile(video_path):
    cmd = f'/home/klab/Desktop/OpenFace/build/bin/FeatureExtraction -f "{video_path}" -out_dir "{OUTPUT_DIRECTORY}" -aus -2Dfp -3Dfp -pdmparams -pose -gaze'
    subprocess.run(cmd, shell=True)


    # Remove unnecessary txt file if it exists
    if os.path.exists(txt_path):
      os.remove(txt_path)
    
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


