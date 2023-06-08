import os
import signal
import nemo.collections.asr as nemo_asr
import json
import shutil
from utilsNEMO import extract_audio_from_mp4

PRETRAINED_MODEL_NAME = "stt_en_conformer_transducer_xlarge"

# # Example usage:
# mp4_path = "inputs/Demo_Video.mp4"
# wav_path = 'inputs_speech/demo_chunks'
# os.makedirs(wav_path, exist_ok=True)
# extract_audio_from_mp4(mp4_path, wav_path)

# manifest = os.path.join(wav_path, 'metadata.json')
# out_path = mp4_path[:-4] + '_out.json'

# # run the evaluation in offline mode
# os.system(f"python scripts_nemo_asr/speech_to_text_eval.py \
#     pretrained_name={PRETRAINED_MODEL_NAME} \
#     dataset_manifest={manifest} \
#     output_filename={out_path} \
#     batch_size=32 \
#     amp=True \
#     use_cer=False")



import os
import subprocess
import time
import datetime
"""

Full Pipeline - NVIDIA NeMo for Automated Speech Recognition (ASR)

NOTE: Make sure to re-run setup.py before using this pipeline. There are LOTS of dependencies!

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('inputs/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_NeMo_ASR/')
WAV_PATH = os.path.abspath('temporary_wav_saving/')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# For timing estimation
valid_videos = [vid for vid in all_videos if os.path.isfile(os.path.join(VIDEO_DIRECTORY, vid))]
unprocessed_videos = [vid for vid in valid_videos if not(os.path.exists(os.path.join(OUTPUT_DIRECTORY, vid[:-4] + '.csv') ))]
num_vids = len(unprocessed_videos)
start_time = time.time()

# Loop through all videos
for i in all_videos:
  save_file = os.path.join(OUTPUT_DIRECTORY, i[:-4] + '.json') 
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  if os.path.exists(save_file):
    print(f'Skipping Video {i}: Output File Already Exists!')
  elif os.path.isfile(video_path):
    os.makedirs(WAV_PATH, exist_ok=True)
    extract_audio_from_mp4(os.path.abspath(video_path), os.path.abspath(WAV_PATH))
    
    manifest = os.path.join(WAV_PATH, 'metadata.json')

    os.system(f"python scripts_nemo_asr/speech_to_text_eval.py \
        pretrained_name={PRETRAINED_MODEL_NAME} \
        dataset_manifest={manifest} \
        output_filename={save_file} \
        batch_size=32 \
        amp=True \
        use_cer=False")
    
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

  

  

