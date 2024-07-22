import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S23_199"
DATETIME_CSV = os.path.abspath(f'/home/klab/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/Raw CSVs/videoFileTable_S{PAT_NOW[4:]}.csv')
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

# Load datetime CSV
datetime_df = pd.read_csv(DATETIME_CSV)
datetime_df['Filename'] = datetime_df['Filename'].str.replace('.m2t', '.mp4')
datetime_df['VideoStart'] = pd.to_datetime(datetime_df['VideoStart'], format='%d-%b-%Y %H:%M:%S')
datetime_df['VideoEnd'] = pd.to_datetime(datetime_df['VideoEnd'], format='%d-%b-%Y %H:%M:%S')

# Function to calculate event start times within the video
def calculate_event_times_within_video(frames, fps=5):
    seconds = np.array(frames) / fps
    minutes = np.floor(seconds / 60).astype(int)
    seconds = np.round(seconds % 60, 1)
    return minutes, seconds

# Function to detect events
def detect_events(emotion_df, au_df):
    events = []

    for emotion, threshold in EVENT_THRESHOLDS.items():
        emotion_frames = emotion_df[emotion_df[emotion] >= threshold]['frame']
        if len(emotion_frames) < MIN_EVENT_LENGTH:
            continue

        # Identify contiguous frames as events
        diff = emotion_frames.diff().fillna(1)
        event_starts = emotion_frames[diff > 1].values
        event_ends = np.append(event_starts[1:] - 1, emotion_frames.iloc[-1])

        for start_frame, end_frame in zip(event_starts, event_ends):
            event_length = end_frame - start_frame + 1
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                events[-1]['End Frame'] = end_frame
                events[-1]['Duration in Seconds'] = round(events[-1]['Duration in Seconds'] + event_length / 5.0, 1)
                continue

            minutes, seconds = calculate_event_times_within_video([start_frame])
            start_time = f"{minutes[0]:02}:{seconds[0]:04.1f}"
            duration = round((end_frame - start_frame + 1) / 5.0, 1)

            avg_au = au_df[(au_df['frame'] >= start_frame) & (au_df['frame'] <= end_frame)].mean()
            avg_emotion = emotion_df[(emotion_df['frame'] >= start_frame) & (emotion_df['frame'] <= end_frame)].mean()

            event_data = {
                'Filename': video_file,
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

for _, row in tqdm(datetime_df.iterrows(), total=len(datetime_df)):
    video_file = row['Filename']
    video_start = row['VideoStart']
    video_end = row['VideoEnd']

    # Load emotion and AU CSVs
    emotion_csv_path = os.path.join(CSV_DIRECTORY, video_file.replace('.mp4', ''), 'outputs_hse.csv')
    au_csv_path = os.path.join(CSV_DIRECTORY, video_file.replace('.mp4', ''), 'outputs_ogau.csv')

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
