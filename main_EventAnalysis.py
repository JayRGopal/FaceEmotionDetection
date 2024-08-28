import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S23_199"
FACEDX_CSV_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_CSV = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'combined_events_{PAT_NOW}.csv')

META_DATA_CSV_PATH = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'chosen_thresholds_{PAT_NOW}.csv')

# Load the thresholds from the meta data CSV
thresholds_df = pd.read_csv(META_DATA_CSV_PATH)

EVENT_THRESHOLDS = dict(zip(thresholds_df['Emotion'], thresholds_df['Threshold']))

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events

FACEDX_FPS = 5  # FPS after downsampling
VIDEO_FPS = 30  # FPS of original video (for time stamps!)

# Function to detect events
def detect_events(emotion_df, au_df):
    events = []

    for emotion, threshold in EVENT_THRESHOLDS.items():
        emotion_values = emotion_df[emotion].values
        frames = emotion_df['frame'].values

        # Identify start and end frames of events
        above_threshold = emotion_values >= threshold
        start_indices = np.where((above_threshold[:-1] == False) & (above_threshold[1:] == True))[0] + 1
        end_indices = np.where((above_threshold[:-1] == True) & (above_threshold[1:] == False))[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            continue

        if start_indices[0] > end_indices[0]:
            end_indices = end_indices[1:]
        if start_indices[-1] > end_indices[-1]:
            start_indices = start_indices[:-1]

        merged_start, merged_end = start_indices[0], end_indices[0]

        for i in range(1, len(start_indices)):
            start_frame, end_frame = frames[start_indices[i]], frames[end_indices[i]]

            if start_frame - frames[merged_end] <= MERGE_TIME:
                # Merge this event with the previous one
                merged_end = end_indices[i]
            else:
                # Save the previous merged event
                event_length = frames[merged_end] - frames[merged_start] + 1
                if event_length >= MIN_EVENT_LENGTH:
                    minutes = int((frames[merged_start] // VIDEO_FPS) // 60)
                    seconds = int((frames[merged_start] // VIDEO_FPS) - (60 * minutes))
                    if minutes >= 60:
                        hours = int(minutes // 60)
                        minutes = minutes % 60
                        start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        start_time = f"{minutes:02d}:{seconds:02d}"

                    event_rows = emotion_df[(emotion_df['frame'] >= frames[merged_start]) & (emotion_df['frame'] <= frames[merged_end])].copy()
                    event_rows['Start Time'] = start_time
                    event_rows['Event Type'] = emotion

                    au_rows = au_df[(au_df['frame'] >= frames[merged_start]) & (au_df['frame'] <= frames[merged_end])].drop(['timestamp', 'success'], axis=1)
                    event_rows = event_rows.merge(au_rows, left_on='frame', right_on='frame', suffixes=('', '_au'))

                    events.append(event_rows)

                # Start a new event
                merged_start, merged_end = start_indices[i], end_indices[i]

        # Save the last merged event
        event_length = frames[merged_end] - frames[merged_start] + 1
        if event_length >= MIN_EVENT_LENGTH:
            minutes = int((frames[merged_start] // VIDEO_FPS) // 60)
            seconds = int((frames[merged_start] // VIDEO_FPS) - (60 * minutes))
            if minutes >= 60:
                hours = int(minutes // 60)
                minutes = minutes % 60
                start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                start_time = f"{minutes:02d}:{seconds:02d}"

            event_rows = emotion_df[(emotion_df['frame'] >= frames[merged_start]) & (emotion_df['frame'] <= frames[merged_end])].copy()
            event_rows['Start Time'] = start_time
            event_rows['Event Type'] = emotion

            au_rows = au_df[(au_df['frame'] >= frames[merged_start]) & (au_df['frame'] <= frames[merged_end])].drop(['timestamp', 'success'], axis=1)
            event_rows = event_rows.merge(au_rows, left_on='frame', right_on='frame', suffixes=('', '_au'))

            events.append(event_rows)

    return pd.concat(events, ignore_index=True) if events else pd.DataFrame()

# Process each video file
all_events = []

# Loop through the first 5 subfolders in the given CSV directory
for subfolder in tqdm(os.listdir(FACEDX_CSV_DIRECTORY)[:5]):
    video_file = subfolder
    
    # Load emotion and AU CSVs
    emotion_csv_path = os.path.join(FACEDX_CSV_DIRECTORY, subfolder, 'outputs_hse.csv')
    au_csv_path = os.path.join(FACEDX_CSV_DIRECTORY, subfolder, 'outputs_ogau.csv')

    if not os.path.exists(emotion_csv_path) or not os.path.exists(au_csv_path):
        print(f"Skipping {video_file}: missing CSV files.")
        continue

    if os.path.getsize(emotion_csv_path) == 0 or os.path.getsize(au_csv_path) == 0:
        print(f"Skipping {video_file}: empty CSV files.")
        continue

    try:
        emotion_df = pd.read_csv(emotion_csv_path)
        au_df = pd.read_csv(au_csv_path
