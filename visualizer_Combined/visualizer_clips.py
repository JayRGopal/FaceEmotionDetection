import cv2
import pandas as pd
import os

def save_event_clips(input_folder, event_analysis_csv, clips_output_folder):
    # Load the event analysis CSV
    events_df = pd.read_csv(event_analysis_csv)
    
    # Ensure the output folder exists
    os.makedirs(clips_output_folder, exist_ok=True)
    
    # Add a new column for clip names
    events_df['Clip Name'] = ''
    
    for index, event in events_df.iterrows():
        video_file = event['Filename']
        video_path = os.path.join(input_folder, video_file)
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate start and end frames
        start_time = event['Start Time'].split(':')
        if len(start_time) == 3:
            start_frame = (int(start_time[0]) * 3600 + int(start_time[1]) * 60 + int(start_time[2])) * fps
        elif len(start_time) == 2:
            start_frame = (int(start_time[0]) * 60 + int(start_time[1])) * fps
        else:
            raise ValueError(f"Unexpected time format: {event['Start Time']}")
        duration = event['Duration in Seconds']
        end_frame = start_frame + (duration * fps)
        
        # Add 2 seconds buffer before and after
        buffer_frames = 2 * fps
        start_frame = max(0, start_frame - buffer_frames)
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), end_frame + buffer_frames)
        
        # Set up clip writer
        clip_name = f"{event['Emotion']}_{index+1}.mp4"
        clip_path = os.path.join(clips_output_folder, clip_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        # Write frames to clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(int(end_frame - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            clip_writer.write(frame)
        
        clip_writer.release()
        cap.release()
        
        # Update the clip name in the DataFrame
        events_df.at[index, 'Clip Name'] = clip_name
    
    cv2.destroyAllWindows()
    
    # Save the updated DataFrame back to CSV
    events_df.to_csv(event_analysis_csv, index=False)

# Usage:
PAT_NOW = 'S20_150'
INPUT_FOLDER = f'/home/jgopal/NAS/Analysis/MP4/{PAT_NOW}_MP4'
CLIPS_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/{PAT_NOW}/'
EVENT_ANALYSIS_CSV = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'combined_events_{PAT_NOW}.csv')

save_event_clips(INPUT_FOLDER, EVENT_ANALYSIS_CSV, CLIPS_OUTPUT_FOLDER)
