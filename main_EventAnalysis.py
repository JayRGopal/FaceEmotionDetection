import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parameters
PAT_NOW = "S20_150"
FACEDX_CSV_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/')
OPENFACE_CSV_DIRECTORY = os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_OpenFace/{PAT_NOW}/')
OUTPUT_CSV = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'combined_events_{PAT_NOW}.csv')

META_DATA_CSV_PATH = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'chosen_thresholds_{PAT_NOW}.csv')

# Load the thresholds from the meta data CSV
thresholds_df = pd.read_csv(META_DATA_CSV_PATH)

EVENT_THRESHOLDS = dict(zip(thresholds_df['Emotion'], thresholds_df['Threshold']))


MIN_EVENT_LENGTH = 2  # Minimum length of each event in frames
MERGE_TIME = 3  # Maximum frames apart to consider merging events

FACEDX_FPS = 5 # FPS after down sampling
VIDEO_FPS = 30  # FPS of original video (for time stamps!)

# Function to detect events
def detect_events(emotion_df, au_df, openface_df):
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

            minutes = int((start_frame // VIDEO_FPS) // 60)
            seconds = int((start_frame // VIDEO_FPS) - (60 * minutes))
            if minutes >= 60:
                hours = int(minutes // 60)
                minutes = minutes % 60
                start_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                start_time = f"{minutes:02d}:{seconds:02d}"
            
            duration = round(event_length / FACEDX_FPS, 1)

            avg_au = au_df[(au_df['frame'] >= start_frame) & (au_df['frame'] <= end_frame)].mean()
            avg_emotion = emotion_df[(emotion_df['frame'] >= start_frame) & (emotion_df['frame'] <= end_frame)].mean()
            avg_openface = openface_df[(openface_df['frame'] >= start_frame) & (openface_df['frame'] <= end_frame)][['AU45_r', 'AU45_c']].mean()
            avg_openface = avg_openface.rename(index={'AU45_r': 'OpenFace_AU45_r', 'AU45_c': 'OpenFace_AU45_c'})

            event_data = {
                'Filename': video_file,
                'Start Time': start_time,
                'Duration in Seconds': duration,
                'Event Type': emotion,
                'End Frame': end_frame
            }
            event_data.update(avg_au.drop(['frame', 'timestamp', 'success']).to_dict())
            event_data.update(avg_emotion.drop(['frame', 'timestamp', 'success']).to_dict())
            event_data.update(avg_openface.to_dict())
            
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
    openface_csv_path = os.path.join(OPENFACE_CSV_DIRECTORY, f'{subfolder[:-4]}.csv')

    if not os.path.exists(emotion_csv_path) or not os.path.exists(au_csv_path):
        print(f"Skipping {video_file}: missing CSV files.")
        continue

    if os.path.getsize(emotion_csv_path) == 0 or os.path.getsize(au_csv_path) == 0:
        print(f"Skipping {video_file}: empty CSV files.")
        continue

    if os.path.getsize(openface_csv_path) == 0:
        print(f"Skipping {video_file}: empty OpenFace CSV file.")
        continue

    try:
        emotion_df = pd.read_csv(emotion_csv_path)
        au_df = pd.read_csv(au_csv_path)
        openface_df = pd.read_csv(openface_csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {video_file}: empty CSV files.")
        continue
    except OSError as e:
        print(f"Skipping {video_file}: OSError - {e}")
        continue

    # Detect events in the video
    video_events = detect_events(emotion_df, au_df, openface_df)

    all_events.extend(video_events)

# Remove 'End Frame' before saving
for event in all_events:
    event.pop('End Frame', None)

# Save all events to a single CSV file
events_df = pd.DataFrame(all_events)
events_df.to_csv(OUTPUT_CSV, index=False)

print(f"Events saved to {OUTPUT_CSV}")
