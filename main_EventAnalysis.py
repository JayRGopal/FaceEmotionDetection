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
def detect_events(emotion_df, au_df, video_file):
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

        for start_frame, end_frame in zip(frames[start_indices], frames[end_indices]):
            event_length = end_frame - start_frame + 1
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                events[-1]['End Frame'] = end_frame
                # Calculate duration as the total timespan of the event
                events[-1]['Duration in Seconds'] = round((end_frame - events[-1]['Start Frame']) / VIDEO_FPS, 1)
                continue

            minutes = int((start_frame // VIDEO_FPS) // 60)
            seconds = int((start_frame // VIDEO_FPS) - (60 * minutes))
            if minutes >= 60:
                hours = int(minutes // 60)
                minutes = minutes % 60
                start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                start_time = f"{minutes:02d}:{seconds:02d}"

            duration = round(event_length / FACEDX_FPS, 1)

            event_data = {
                'Filename': video_file,
                'Start Time': start_time,
                'Duration in Seconds': duration,
                'Event Type': emotion,
                'Start Frame': start_frame,
                'End Frame': end_frame
            }

            events.append(event_data)

    return events

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
        au_df = pd.read_csv(au_csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {video_file}: empty CSV files.")
        continue
    except OSError as e:
        print(f"Skipping {video_file}: OSError - {e}")
        continue

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df, video_file)

    if video_events:
        all_events.extend(video_events)

# Remove 'End Frame' and 'Start Frame' before saving
for event in all_events:
    event.pop('End Frame', None)
    event.pop('Start Frame', None)

# Convert to DataFrame
all_events_df = pd.DataFrame(all_events)

# Add Clip Name column
global_event_counter = 0
clip_names = []
previous_key = None

for _, row in all_events_df.iterrows():
    key = (row['Start Time'], row['Filename'])
    
    # Increment the global event counter only when encountering a new key
    if key != previous_key:
        global_event_counter += 1
        previous_key = key
    
    clip_name = f"{row['Event Type']}_{global_event_counter}.mp4"
    clip_names.append(clip_name)

# Assign the Clip Name column to the DataFrame
all_events_df['Clip Name'] = clip_names

# Reorder columns
meta_columns = ['Clip Name', 'Start Time', 'Filename', 'Event Type', 'Duration in Seconds']
all_columns = meta_columns + [col for col in all_events_df.columns if col not in meta_columns]
all_events_df = all_events_df[all_columns]

# Save all events to a single CSV file
all_events_df.to_csv(OUTPUT_CSV, index=False)
print(f"Events saved to {OUTPUT_CSV}")
