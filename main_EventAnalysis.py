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
thresholds_df.columns = thresholds_df.columns.str.strip()  # Clean column names

EVENT_THRESHOLDS = dict(zip(thresholds_df['Emotion'], thresholds_df['Threshold']))

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events

FACEDX_FPS = 5  # FPS after down sampling
VIDEO_FPS = 30  # FPS of original video (for time stamps!)

# Function to clean DataFrame column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

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
        
        for start_frame, end_frame in zip(frames[start_indices], frames[end_indices]):
            event_length = end_frame - start_frame + 1
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                events[-1]['End Frame'] = end_frame
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

# Loop through the subfolders in the given CSV directory
for subfolder in tqdm(os.listdir(FACEDX_CSV_DIRECTORY)):
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
        emotion_df = clean_column_names(emotion_df)
        au_df = pd.read_csv(au_csv_path)
        au_df = clean_column_names(au_df)
    except pd.errors.EmptyDataError:
        print(f"Skipping {video_file}: empty CSV files.")
        continue
    except OSError as e:
        print(f"Skipping {video_file}: OSError - {e}")
        continue

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df)

    for event in video_events:
        start_frame = event['Start Frame']
        end_frame = event['End Frame']

        # Get the frame-by-frame data for this event
        event_au_df = au_df[(au_df['frame'] >= start_frame) & (au_df['frame'] <= end_frame)]
        event_emotion_df = emotion_df[(emotion_df['frame'] >= start_frame) & (emotion_df['frame'] <= end_frame)]

        # Merge AU and emotion data
        event_data = pd.merge(event_au_df, event_emotion_df.drop(columns=['timestamp', 'success']), on='frame')

        # Add event metadata to each row
        event_data['Filename'] = video_file
        event_data['Start Time'] = event['Start Time']
        event_data['Duration in Seconds'] = event['Duration in Seconds']
        event_data['Event Type'] = event['Event Type']

        all_events.append(event_data)

# Concatenate all event data
if all_events:
    events_df = pd.concat(all_events, ignore_index=True)

    # Clean column names of the final DataFrame
    events_df = clean_column_names(events_df)

    # Add Clip Name column
    clip_name_list = []
    clip_index = 1
    current_event_type = events_df.iloc[0]['Event Type']

    for i, row in events_df.iterrows():
        if row['Event Type'] != current_event_type:
            clip_index += 1
            current_event_type = row['Event Type']
        clip_name_list.append(f"{row['Event Type']}_{clip_index}.mp4")

    events_df.insert(0, 'Clip Name', clip_name_list)  # Insert at the very left

    # Reorder columns so that meta columns are first
    meta_columns = ['Clip Name', 'Filename', 'Start Time', 'Duration in Seconds', 'Event Type']
    other_columns = [col for col in events_df.columns if col not in meta_columns]
    ordered_columns = meta_columns + other_columns
    events_df = events_df[ordered_columns]

    # Save all events to a single CSV file
    events_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Events saved to {OUTPUT_CSV}")
else:
    print("No events to save.")
