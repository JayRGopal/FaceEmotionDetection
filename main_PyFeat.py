from feat import Detector
import os
import gc
import cv2
import pandas as pd

# Main parameters
PAT_NOW = "S23_199"
VIDEO_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/MP4/{PAT_NOW}_MP4')
OUTPUT_DIRECTORY_PYFEAT = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_PyFeat/{PAT_NOW}/')

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

# Loop through all videos and process each frame with detect_aus
for video_name in unprocessed_videos:
    video_path = os.path.join(VIDEO_DIRECTORY, video_name)
    
    # Create a subfolder for each video inside the PyFeat output directory
    save_folder = SAVE_PATH_FOLDER(video_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Open video and set up frame skipping
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 5)  # Down-sample to 5 fps
    
    all_frames_data = []
    frame_index = 0
    
    print(f'Processing frames for {video_path}...')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_interval == 0:
            # Detect AUs for the current frame without landmarks
            aus_data = detector.detect_aus(frame)
            
            # Append AU data to list (convert to DataFrame for ease)
            aus_df = pd.DataFrame([aus_data])
            all_frames_data.append(aus_df)
        
        frame_index += 1
    
    cap.release()
    
    # Combine all frame data into a single DataFrame
    if all_frames_data:
        pyfeat_output = pd.concat(all_frames_data, ignore_index=True)
    
        # Save the output to CSV
        save_path = SAVE_PATH_PYFEAT(video_name)
        pyfeat_output.to_csv(save_path, index=False)
        print(f'Saved PyFeat CSV to {save_path}!')
    
    # Clear memory after processing each video
    gc.collect()

print("Processing complete!")
