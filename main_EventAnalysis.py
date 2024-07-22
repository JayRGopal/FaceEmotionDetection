import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S23_199"
CSV_DIRECTORY = os.path.abspath(f'/home/klab/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_CSV = os.path.join(CSV_DIRECTORY, 'combined_events.csv')

EVENT_THRESHOLDS = {
    'Happiness': 0.9,
    'Anger': 0.85,
    'Sadness': 0.9,
    'Neutral': 0.95
}

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events
FACEDX_FPS = 5 # FPS after down sampling
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

        for start_frame, end_frame in zip(frames[start_indices], frames[end_indices]):
            event_length = end_frame - start_frame + 1
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                events[-1]['End Frame'] = end_frame
                events[-1]['Duration in Seconds'] = round(events[-1]['Duration in Seconds'] + event_length / FACEDX_FPS, 1)
                continue

            minutes = (start_frame // VIDEO_FPS) // 60
            seconds = (start_frame // VIDEO_FPS) - (60 * minutes)
            start_time = f"{minutes}:{seconds}"
            duration = round(event_length / FACEDX_FPS, 1)

            avg_au = au_df[(au_df['frame'] >= start_frame) & (au_df['frame'] <= end_frame)].mean()
            avg_emotion = emotion_df[(emotion_df['frame'] >= start_frame) & (emotion_df['frame'] <= end_frame)].mean()

            event_data = {
                'Filename': video_file,
                'Start Frame Num': start_frame,
                'Start Time': start_time,
                'Duration in Seconds': duration,
                'Event Type': emotion,
                'End Frame': end_frame
            }
            event_data.update(avg_au.drop(['frame', 'timestamp', 'success']).to_dict())
            event_data.update(avg_emotion.drop(['frame', 'timestamp', 'success']).to_dict())

            events.append(event_data)

    return events

# Process each video file
all_events = []

# Loop through the subfolders in the given CSV directory
for subfolder in tqdm(os.listdir(CSV_DIRECTORY)[10:20]):
    video_file = subfolder + '.mp4'
    
    # Load emotion and AU CSVs
    emotion_csv_path = os.path.join(CSV_DIRECTORY, subfolder, 'outputs_hse.csv')
    au_csv_path = os.path.join(CSV_DIRECTORY, subfolder, 'outputs_ogau.csv')

    if not os.path.exists(emotion_csv_path) or not os.path.exists(au_csv_path):
        continue

    emotion_df = pd.read_csv(emotion_csv_path)
    au_df = pd.read_csv(au_csv_path)

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df)
    
    if video_events:  # Debugging: Check if events are detected
        print(f"Events detected in {video_file}: {len(video_events)}")
    else:
        print(f"No events detected in {video_file}")

    all_events.extend(video_events)

# Remove 'End Frame' before saving
for event in all_events:
    event.pop('End Frame', None)

# Save all events to a single CSV file
events_df = pd.DataFrame(all_events)
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Events saved to {OUTPUT_CSV}")
