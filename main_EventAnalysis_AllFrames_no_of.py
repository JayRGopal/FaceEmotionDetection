import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S23_199"
user_name = "bkakusa"
FACEDX_CSV_DIRECTORY = os.path.abspath(f'/home/{user_name}/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_DIRECTORY = os.path.abspath(f'/home/{user_name}/NAS/Analysis/outputs_EventAnalysis/')
OUTPUT_CSV = os.path.join(OUTPUT_DIRECTORY, f'combined_events_{PAT_NOW}.csv')

META_DATA_CSV_PATH = os.path.join(os.path.abspath(f'/home/{user_name}/NAS/Analysis/outputs_EventAnalysis/'), f'chosen_thresholds_{PAT_NOW}.csv')


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

        # Identify events and get indices
        above_threshold = emotion_values >= threshold
        event_indices = np.nonzero(above_threshold)
        event_indices = event_indices[0]

        # Skip if too few events found
        if len(event_indices) < MIN_EVENT_LENGTH:
            continue

        # Merge events closeby

        # Identify events in list that are within MERGE_TIME
        merge_threshold = np.diff(event_indices)
        merge_threshold = np.concatenate((merge_threshold, np.array([merge_threshold[-1]])))
        merge_threshold = (merge_threshold < MERGE_TIME)*1


        # Add start index of first event
        start_indices = [event_indices[0]]
        end_indices = []
        last_value = merge_threshold[0]

        #If event is stand alone, add end index
        if last_value == 0:
            end_indices.append(event_indices[0])

        # Iterate through events and mark start and end times, merging as you go

        for merge_idx, merge_value in enumerate(merge_threshold):
            if merge_idx == 0: # Skip first index as already accounted for
                continue
            elif last_value == 1 and merge_value == last_value: # If last and merge values are 1, merge events
                continue
            elif last_value == 1 and merge_value == 0: # If merge value 0 when last value was 1, then at end of merging, add end index and also add this standalone event
                end_indices.append(event_indices[merge_idx])
            elif merge_value == 0: # If merge value and last value are 0, add stand alone event
                start_indices.append(event_indices[merge_idx])
                end_indices.append(event_indices[merge_idx])
            elif last_value == 0 and merge_value == 1: # If merge value is 1 and last is 0, start of new merge
                start_indices.append(event_indices[merge_idx])
            last_value = merge_value # Keep track of last value

        # NEED TO FIX SO THAT END EVENTS BETWEEN FILES MERGE IF INDICATED
        if len(start_indices) > len(end_indices) and last_value == 1:
            loop_event = 1
            end_indices.append(event_indices[-1])


        # Remove Events Less than MIN_EVENT_LENGTH
        end_indices = np.array(end_indices)
        start_indices = np.array(start_indices)
        event_length = end_indices - start_indices
        end_indices = end_indices[event_length >= MIN_EVENT_LENGTH - 1]
        start_indices = start_indices[event_length >= MIN_EVENT_LENGTH - 1]

        # Skip if no more events
        if len(start_indices) == 0:
            continue


        # Ensure lists are same size and in correct order
        assert len(start_indices) == len(end_indices)
        assert start_indices[0] < end_indices[0]
        assert start_indices[-1] < end_indices[-1]

        # Get event start time stamps
        minutes = np.floor_divide(np.floor_divide(frames[start_indices], VIDEO_FPS), 60)
        seconds = np.floor_divide(frames[start_indices], VIDEO_FPS) - (60 * minutes)
        hours = np.floor_divide(minutes, 60)

        minutes = minutes.astype(int)
        seconds = seconds.astype(int)
        hours = hours.astype(int)

        start_times = []
        for hour_val, minute_val, second_val in zip(hours, minutes, seconds):
            start_times.append( f"{hour_val:02d}:{minute_val:02d}:{second_val:02d}")

        # Get event durations
        durations = np.round(np.divide(end_indices - start_indices + 1, FACEDX_FPS), 1)

        # Convert frame columns to integers for consistent comparison
        au_df['frame'] = au_df['frame'].astype(int)
        emotion_df['frame'] = emotion_df['frame'].astype(int)

        # Iterate through each event to add inbetween frames
        for event_idx, _ in enumerate(start_indices):
            event_data = {
                'Filename': video_file,
                'Start Time': start_times[event_idx],
                'Duration in Seconds': durations[event_idx],  # Ensure duration is set initially
                'Event Type': emotion,
                'End Frame': frames[end_indices[event_idx]],
            }
            for frame_idx in range(start_indices[event_idx], end_indices[event_idx] + 1):

                # Use integer comparison for frames
                frame_au = au_df[au_df['frame'] == int(frames[frame_idx])].drop(['frame', 'timestamp', 'success'], axis=1)
                frame_emotion = emotion_df[emotion_df['frame'] == int(frames[frame_idx])].drop(['frame', 'timestamp', 'success'],axis=1)

                # Check if the frame exists in the data, otherwise skip this frame
                if frame_au.empty or frame_emotion.empty:
                    continue

                frame_data = event_data.copy()
                frame_data['Frame'] = frames[frame_idx]
                frame_data['Frame Num'] = (frame_idx - start_indices[event_idx]) + 1
                frame_data['Time'] = np.round(np.divide((frame_data['Frame Num'] - 1), FACEDX_FPS), 1)
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


# Step 1: Assign each event a sequential number based on its order in the DataFrame (not by event type)
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

# Add the "Clip Name" column to the DataFrame
events_df['Clip Name'] = clip_values

# Save the updated DataFrame back to the CSV
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Post-processing complete. Events with corrected durations saved to {OUTPUT_CSV}")