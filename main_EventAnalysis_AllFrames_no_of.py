import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S23_199"
FACEDX_CSV_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/')
OUTPUT_CSV = os.path.join(OUTPUT_DIRECTORY, f'combined_events_{PAT_NOW}.csv')

META_DATA_CSV_PATH = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'chosen_thresholds_{PAT_NOW}.csv')


# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Load the thresholds from the meta data CSV
thresholds_df = pd.read_csv(META_DATA_CSV_PATH)

EVENT_THRESHOLDS = dict(zip(thresholds_df['Emotion'], thresholds_df['Threshold']))

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events

FACEDX_FPS = 5  # FPS after down sampling
VIDEO_FPS = 30  # FPS of original video (for time stamps!)

# Function to detect and merge events
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
            start_frame = int(start_frame)  # Ensure it's an integer
            end_frame = int(end_frame)      # Ensure it's an integer
            # Calculate event length based on the available rows in hse or ogau CSV
            event_length = len(au_df[(au_df['frame'] >= start_frame) & (au_df['frame'] <= end_frame)])
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                # Ensure 'Duration in Seconds' exists before updating it
                if 'Duration in Seconds' not in events[-1]:
                    events[-1]['Duration in Seconds'] = 0

                events[-1]['End Frame'] = end_frame
                events[-1]['Duration in Seconds'] = round(events[-1]['Duration in Seconds'] + event_length / FACEDX_FPS, 1)

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
                'Duration in Seconds': duration,  # Ensure duration is set initially
                'Event Type': emotion,
                'End Frame': end_frame
            }

            # Add the full range of frames (at 5 FPS) for this event
            for frame in range(start_frame, end_frame + 1):
                # Convert frame columns to integers for consistent comparison
                au_df['frame'] = au_df['frame'].astype(int)
                emotion_df['frame'] = emotion_df['frame'].astype(int)

                # Use integer comparison for frames
                frame_au = au_df[au_df['frame'] == int(frame)].drop(['frame', 'timestamp', 'success'], axis=1)
                frame_emotion = emotion_df[emotion_df['frame'] == int(frame)].drop(['frame', 'timestamp', 'success'], axis=1)

                # Check if the frame exists in the data, otherwise skip this frame
                if frame_au.empty or frame_emotion.empty:
                    continue

                
                frame_data = event_data.copy()
                frame_data['Frame'] = frame
                frame_data.update(frame_au.to_dict(orient='records')[0] if not frame_au.empty else {})
                frame_data.update(frame_emotion.to_dict(orient='records')[0] if not frame_emotion.empty else {})

                events.append(frame_data)

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
        au_df = pd.read_csv(au_csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {video_file}: empty CSV files.")
        continue
    except OSError as e:
        print(f"Skipping {video_file}: OSError - {e}")
        continue

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df)

    all_events.extend(video_events)

# Remove 'End Frame' before saving
for event in all_events:
    event.pop('End Frame', None)

# Save all events to a single CSV file
events_df = pd.DataFrame(all_events)
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Events saved to {OUTPUT_CSV}")

# Load the saved CSV into a DataFrame for post-processing
events_df = pd.read_csv(OUTPUT_CSV)
import pdb; pdb.set_trace()
# Step 1: Check for duration discrepancies in the last frame of each event
# Loop through the events to adjust any incorrect durations
for i in range(1, len(events_df)):
    # Check if the event is the same as the previous row
    if (events_df.loc[i, 'Filename'] == events_df.loc[i-1, 'Filename']) and \
       (events_df.loc[i, 'Start Time'] == events_df.loc[i-1, 'Start Time']):
        # If the duration is different, correct it to match the previous row's duration
        if events_df.loc[i, 'Duration in Seconds'] != events_df.loc[i-1, 'Duration in Seconds']:
            events_df.loc[i, 'Duration in Seconds'] = events_df.loc[i-1, 'Duration in Seconds']
import pdb; pdb.set_trace()
# Step 2: Assign each event a sequential number based on its order in the DataFrame (not by event type)
clip_values = []
event_counter = 1  # Start a global counter across all events

# Iterate over the rows to assign a sequential number to each event
for i in range(len(events_df)):
    event_type = events_df.loc[i, 'Event Type']
    
    # Create the Clip name in the format {Event Type}_{Number}.mp4
    clip_name = f"{event_type}_{event_counter}.mp4"
    clip_values.append(clip_name)
    
    # Increment the counter for each new event
    # Check for event change by comparing the current and next rows
    if i == len(events_df) - 1 or \
       events_df.loc[i, 'Filename'] != events_df.loc[i+1, 'Filename'] or \
       events_df.loc[i, 'Start Time'] != events_df.loc[i+1, 'Start Time']:
        event_counter += 1

# Add the "Clip" column to the DataFrame
events_df['Clip'] = clip_values

# Save the updated DataFrame back to the CSV
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Post-processing complete. Events with corrected durations saved to {OUTPUT_CSV}")