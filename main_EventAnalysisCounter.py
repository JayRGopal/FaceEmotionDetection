import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S20_150"
CSV_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_COUNTS_CSV = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'event_detection_counts_{PAT_NOW}.csv')

# Thresholds to test
THRESHOLDS = np.arange(0.5, 1.05, 0.05)

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events

FACEDX_FPS = 5 # FPS after down sampling
VIDEO_FPS = 30  # FPS of original video (for time stamps!)

# Function to detect events
def detect_events(emotion_df, threshold, emotion):
    events = []
    emotion_values = emotion_df[emotion].values
    frames = emotion_df['frame'].values

    # Identify start and end frames of events
    above_threshold = emotion_values >= threshold
    start_indices = np.where((above_threshold[:-1] == False) & (above_threshold[1:] == True))[0] + 1
    end_indices = np.where((above_threshold[:-1] == True) & (above_threshold[1:] == False))[0]

    if len(start_indices) == 0 or len(end_indices) == 0:
        return events

    if start_indices[0] > end_indices[0]:
        end_indices = end_indices[1:]
    if start_indices[-1] > end_indices[-1]:
        start_indices = start_indices[:-1]

    for start_frame, end_frame in zip(frames[start_indices], frames[end_indices]):
        event_length = end_frame - start_frame + 1
        if event_length < MIN_EVENT_LENGTH:
            continue

        # Merge close by events
        if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
            events[-1]['End Frame'] = end_frame
            continue

        event_data = {
            'Start Frame': start_frame,
            'End Frame': end_frame,
        }

        events.append(event_data)

    return events

# Initialize a dictionary to hold event counts
event_counts = {emotion: {threshold: 0 for threshold in THRESHOLDS} for emotion in ['Happiness', 'Anger', 'Sadness', 'Neutral']}

# Process each video file
for subfolder in tqdm(os.listdir(CSV_DIRECTORY)):
    video_file = subfolder
    
    # Load emotion and AU CSVs
    emotion_csv_path = os.path.join(CSV_DIRECTORY, subfolder, 'outputs_hse.csv')

    if not os.path.exists(emotion_csv_path):
        print(f"Skipping {video_file}: missing CSV files.")
        continue

    if os.path.getsize(emotion_csv_path) == 0:
        print(f"Skipping {video_file}: empty CSV files.")
        continue

    try:
        emotion_df = pd.read_csv(emotion_csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {video_file}: empty CSV files.")
        continue
    except OSError as e:
        print(f"Skipping {video_file}: OSError - {e}")
        continue

    for emotion in event_counts.keys():
        for threshold in THRESHOLDS:
            events = detect_events(emotion_df, threshold, emotion)
            event_counts[emotion][threshold] += len(events)

# Convert the event_counts dictionary to a DataFrame and save it to a CSV
event_counts_df = pd.DataFrame(event_counts)
event_counts_df.index = THRESHOLDS
event_counts_df.index.name = 'Threshold'

event_counts_df.to_csv(OUTPUT_COUNTS_CSV)

print(f"Event counts saved to {OUTPUT_COUNTS_CSV}")
