from feat import Detector
import os
import gc
import cv2

# Main parameters
PAT_NOW = "S23_199"
VIDEO_DIRECTORY = os.path.abspath(f'/home/klab/NAS/Analysis/MP4/{PAT_NOW}_MP4')
OUTPUT_DIRECTORY_PYFEAT = os.path.abspath(f'/home/klab/NAS/Analysis/outputs_PyFeat/{PAT_NOW}/')

# Create PyFeat output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY_PYFEAT):
    os.makedirs(OUTPUT_DIRECTORY_PYFEAT)

# Function that returns the save folder and path for PyFeat CSV output
SAVE_PATH_FOLDER = lambda video_name: os.path.join(OUTPUT_DIRECTORY_PYFEAT, video_name)
SAVE_PATH_PYFEAT = lambda video_name: os.path.join(SAVE_PATH_FOLDER(video_name), 'outputs_pyfeat.csv')

# List of unprocessed videos
def get_valid_vids(directory):
    return [f for f in os.listdir(directory) if f.endswith(('.mp4', '.m2t'))]

unprocessed_videos = get_valid_vids(VIDEO_DIRECTORY)

# Initialize PyFeat detector
detector = Detector()

# Loop through all videos and process with PyFeat
for video_name in unprocessed_videos:
    video_path = os.path.join(VIDEO_DIRECTORY, video_name)
    
    # Create a subfolder for each video inside the PyFeat output directory
    save_folder = SAVE_PATH_FOLDER(video_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Run PyFeat detection on the video
    print(f'Running PyFeat on {video_path}...')
    pyfeat_output = detector.detect_video(video_path, skip_frames=6)
    
    # Save PyFeat output to CSV
    save_path = SAVE_PATH_PYFEAT(video_name)
    pyfeat_output.to_csv(save_path, index=False)
    print(f'Saved PyFeat CSV to {save_path}!')
    
    # Clear memory after processing each video
    gc.collect()

print("Processing complete!")


