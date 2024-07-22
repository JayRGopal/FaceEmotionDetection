import os
import pandas as pd
import numpy as np
import datetime

# Parameters
PAT_NOW = "S23_199"
DATETIME_CSV = os.path.abspath(f'/home/klab/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/Raw CSVs/videoFileTable_S{PAT_NOW[4:]}.csv')
CSV_DIRECTORY = os.path.abspath(f'/home/klab/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OUTPUT_CSV = os.path.join(CSV_DIRECTORY, 'combined_events.csv')


EVENT_THRESHOLDS = {
    'Happiness': 0.85,
    'Anger': 0.85,
    'Sadness': 0.9,
    'Neutral': 0.9
}

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 5  # Maximum frames apart to consider merging events

# Load datetime CSV
datetime_df = pd.read_csv(DATETIME_CSV)
datetime_df['Filename'] = datetime_df['Filename'].str.replace('.m2t', '.mp4')
datetime_df['VideoStart'] = pd.to_datetime(datetime_df['VideoStart'], format='%d-%b-%Y %H:%M:%S')
datetime_df['VideoEnd'] = pd.to_datetime(datetime_df['VideoEnd'], format='%d-%b-%Y %H:%M:%S')

# Function to calculate start and end times of events
def calculate_event_times(video_start, frames, fps=5):
    start_times = video_start + pd.to_timedelta(np.array(frames) / fps, unit='s')
    return start_times

# Function to detect events
def detect_events(emotion_df, au_df, video_start):
    events = []

    for emotion, threshold in EVENT_THRESHOLDS.items():
        emotion_frames = emotion_df[emotion_df[emotion] >= threshold]['frame']
        if len(emotion_frames) < MIN_EVENT_LENGTH:
            continue

        # Identify contiguous frames as events
        diff = emotion_frames.diff().fillna(1)
        event_starts = emotion_frames[diff > 1].values
        event_ends = emotion_frames[diff > 1].shift(-1).fillna(emotion_frames.iloc[-1]).values

        for start_frame, end_frame in zip(event_starts, event_ends):
            event_length = end_frame - start_frame + 1
            if event_length < MIN_EVENT_LENGTH:
                continue

            # Merge close by events
            if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
                events[-1]['End Frame'] = end_frame
                events[-1]['Duration in Seconds'] += event_length / 5.0
                continue

            start_time = calculate_event_times(video_start, [start_frame])[0]
            end_time = calculate_event_times(video_start, [end_frame])[0]
            duration = (end_time - start_time).total_seconds()

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

for _, row in datetime_df.iterrows():
    video_file = row['Filename']
    video_start = row['VideoStart']
    video_end = row['VideoEnd']

    # Load emotion and AU CSVs
    emotion_csv_path = os.path.join(CSV_DIRECTORY, video_file, 'outputs_hse.csv')
    au_csv_path = os.path.join(CSV_DIRECTORY, video_file, 'outputs_ogau.csv')

    if not os.path.exists(emotion_csv_path) or not os.path.exists(au_csv_path):
        continue

    emotion_df = pd.read_csv(emotion_csv_path)
    au_df = pd.read_csv(au_csv_path)

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df, video_start)
    all_events.extend(video_events)

# Remove 'End Frame' before saving
for event in all_events:
    event.pop('End Frame', None)

# Save all events to a single CSV file
events_df = pd.DataFrame(all_events)
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Events saved to {OUTPUT_CSV}")
