import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
patients_videos = {
    'S23_202': ['1896WZ01', '1896X000', '1896X100', '1896X900', '1896X901'],
    'S23_203': ['1896YR00', '1896YR01'],
    'S23_212': ['61901P01', '61901Q00', '61901R00', '61901R01']
}

FACEDX_CSV_DIRECTORY = '/home/jgopal/NAS/Analysis/outputs_Combined/'
OUTPUT_CSV = '/home/jgopal/NAS/Analysis/outputs_EventAnalysis/selected_happiness_events.csv'

MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events
FACEDX_FPS = 5
VIDEO_FPS = 30
MIN_EVENTS = 10  # Minimum number of happiness events required
MIN_THRESHOLD = 0.05  # Minimum threshold limit to prevent excessive events

def detect_happiness_events(emotion_df, threshold):
    events = []
    emotion_values = emotion_df['Happiness'].values
    frames = emotion_df['frame'].values

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

        if events and start_frame - events[-1]['End Frame'] <= MERGE_TIME:
            events[-1]['End Frame'] = end_frame
            continue

        start_time = f"{int(start_frame // VIDEO_FPS) // 60:02}:{int(start_frame // VIDEO_FPS) % 60:02}"
        end_time = f"{int(end_frame // VIDEO_FPS) // 60:02}:{int(end_frame // VIDEO_FPS) % 60:02}"

        events.append({'Start Time': start_time, 'End Time': end_time, 'End Frame': end_frame})

    return events

# Process specified videos
all_events = []

for patient, videos in patients_videos.items():
    meta_data_path = os.path.join('/home/jgopal/NAS/Analysis/outputs_EventAnalysis/', f'chosen_thresholds_{patient}.csv')
    thresholds_df = pd.read_csv(meta_data_path)
    happiness_threshold = thresholds_df.set_index('Emotion').at['Happiness', 'Threshold']

    # Adjust threshold until we reach at least 10 events or hit the minimum threshold
    total_events = 0
    events_list = []
    while total_events < MIN_EVENTS and happiness_threshold >= MIN_THRESHOLD:
        events_list = []
        for video in tqdm(videos, desc=f"Processing videos for patient {patient} with threshold {happiness_threshold:.2f}"):
            emotion_csv_path = os.path.join(FACEDX_CSV_DIRECTORY, patient, video, 'outputs_hse.csv')
            if not os.path.exists(emotion_csv_path) or os.path.getsize(emotion_csv_path) == 0:
                continue

            try:
                emotion_df = pd.read_csv(emotion_csv_path)
            except pd.errors.EmptyDataError:
                continue

            video_events = detect_happiness_events(emotion_df, happiness_threshold)
            for event in video_events:
                events_list.append({
                    'Patient': patient,
                    'Video': video,
                    'Start Time': event['Start Time'],
                    'End Time': event['End Time']
                })

        total_events = len(events_list)
        if total_events < MIN_EVENTS:
            happiness_threshold -= 0.05
            if happiness_threshold < MIN_THRESHOLD:
                happiness_threshold = MIN_THRESHOLD

    all_events.extend(events_list)

# Save all events to a single CSV file
events_df = pd.DataFrame(all_events)
events_df.to_csv(OUTPUT_CSV, index=False)
print(f"Events saved to {OUTPUT_CSV}")
