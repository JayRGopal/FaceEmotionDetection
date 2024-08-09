import cv2
import pandas as pd
import numpy as np
import os

def visualize_analysis(input_folder, csv_output_folder, video_output_folder, threshold=0.8, emotion_now='Happiness', USE_BBOXES=True, SAVE_CLIPS=False, CLIPS_OUTPUT_FOLDER=None):
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # Get the last part of the input folder path for naming the output video
    output_video_name = os.path.basename(os.path.normpath(input_folder)) + f'_{emotion_now}_{threshold}_output.mp4'
    combined_output_path = os.path.join(video_output_folder, output_video_name)
    
    # Initialize the VideoWriter object later
    out = None
    
    # If saving clips, ensure the output folder exists
    if SAVE_CLIPS and CLIPS_OUTPUT_FOLDER:
        os.makedirs(CLIPS_OUTPUT_FOLDER, exist_ok=True)
    
    def draw_data_emotion(data, start_x, canvas):
        max_items_per_column = 8
        num_cols = ((len(list(data.columns)) - 3) + max_items_per_column - 1 ) // max_items_per_column
        col_width = width // num_cols
        for idx, (col, val) in enumerate(data.drop(columns=['frame', 'timestamp', 'success']).items()):
            color = (0, 0, 255) if val.values[0] > threshold else (0, 0, 0)
            text = f"{col}: {round(val.values[0], 3)}"
            x_offset = (idx // max_items_per_column) * col_width
            y_offset = 50 * (idx % max_items_per_column)  # Adjusted y_offset to accommodate bigger text
            cv2.putText(canvas, text, (start_x + x_offset, 50 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)  # Increased font scale and thickness
    
    def draw_data_au(data, start_x, canvas):
        max_items_per_column = 11
        num_cols = ((len(list(data.columns)) - 3) + max_items_per_column - 1 ) // max_items_per_column
        col_width = width // num_cols
        for idx, (col, val) in enumerate(data.drop(columns=['frame', 'timestamp', 'success']).items()):
            color = (0, 0, 255) if val.values[0] > threshold else (0, 0, 0)
            text = f"{col}: {round(val.values[0], 2)}"
            x_offset = (idx // max_items_per_column) * col_width
            y_offset = 40 * (idx % max_items_per_column)  # Adjusted y_offset to accommodate bigger text
            cv2.putText(canvas, text, (start_x + x_offset, 40 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)  # Increased font scale and thickness
   

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        output_base_path = os.path.splitext(video_file)[0]
        
        au_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_ogau.csv')
        emotion_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_hse.csv')
        bbox_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_bboxes.csv')
        
        if os.path.exists(au_csv_path):
            try:
                # Load CSVs
                au_df = pd.read_csv(au_csv_path)
                emotion_df = pd.read_csv(emotion_csv_path)
                if USE_BBOXES:
                    bbox_df = pd.read_csv(bbox_csv_path)
            except pd.errors.EmptyDataError:
                print(f"Skipping {video_file} due to empty CSV files.")
                continue
        else:
            print(f"Skipping {video_file} due to missing CSV files.")
            continue
        
        # Filter frames based on emotion threshold
        filtered_frames = emotion_df[emotion_df[emotion_now] >= threshold]['frame'].unique()
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if out is None:
            # Initialize VideoWriter if not already done
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(combined_output_path, fourcc, fps, (3*width, height))
        
        au_data_last = None
        emotion_data_last = None
        bbox_data_last = None
        
        # Initialize variables for event detection and saving
        event_frames = []
        event_index = 0
        clip_writer = None
        buffer_frames = 2 * fps  # 2 second buffer
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            if current_frame not in filtered_frames:
                if SAVE_CLIPS and clip_writer is not None:
                    # Add buffer frames after the event
                    for _ in range(buffer_frames):
                        ret, buffer_frame = cap.read()
                        if ret:
                            clip_writer.write(buffer_frame)
                    clip_writer.release()
                    clip_writer = None
                continue
            
            # Check if there's data for the current frame
            au_data = au_df[au_df['frame'] == current_frame]
            emotion_data = emotion_df[emotion_df['frame'] == current_frame]
            if USE_BBOXES:
                bbox_data = bbox_df[bbox_df['frame'] == current_frame]
            
            # If no data for the current frame, use the previous frame's data
            if au_data.empty:
                au_data = au_data_last
            else:
                au_data_last = au_data
            
            if emotion_data.empty:
                emotion_data = emotion_data_last
            else:
                emotion_data_last = emotion_data
            
            if USE_BBOXES:
                if bbox_data.empty:
                    bbox_data = bbox_data_last
                else:
                    bbox_data_last = bbox_data
            
            # Create a copy of the frame for the combined video
            combined_frame = frame.copy()
            
            # Extract facial bounding box coordinates for the current frame
            if USE_BBOXES and not bbox_data.empty:
                x, y, w, h = bbox_data.iloc[0]['Facial Box X'], bbox_data.iloc[0]['Facial Box Y'], bbox_data.iloc[0]['Facial Box W'], bbox_data.iloc[0]['Facial Box H']
                
                # Draw the facial bounding box on the combined frame only
                cv2.rectangle(combined_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Create a blank canvas
            canvas = 255 * np.ones((height, 3*width, 3), dtype=np.uint8)
            
            # Place the combined frame (with bounding box) in the middle
            canvas[:, width:2*width, :] = combined_frame
            
            # Write the emotion outputs on the left
            if emotion_data is not None:
                draw_data_emotion(emotion_data, 10, canvas)
            
            # Write the AU outputs on the right
            if au_data is not None:
                draw_data_au(au_data, 2*width + 10, canvas)
            
            # Write the frame to the output video
            out.write(canvas)
            
            # Save clips if enabled (without bounding box)
            if SAVE_CLIPS:
                if clip_writer is None:
                    # Start of a new event
                    clip_path = os.path.join(CLIPS_OUTPUT_FOLDER, f"{emotion_now.lower()}_{event_index}.mp4")
                    clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                    event_index += 1
                    
                    # Add buffer frames before the event
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - buffer_frames))
                    for _ in range(buffer_frames):
                        ret, buffer_frame = cap.read()
                        if ret:
                            clip_writer.write(buffer_frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                clip_writer.write(frame)  # Write the original frame without bounding box
        
        cap.release()
        if SAVE_CLIPS and clip_writer is not None:
            # Add buffer frames after the last event
            for _ in range(buffer_frames):
                ret, buffer_frame = cap.read()
                if ret:
                    clip_writer.write(buffer_frame)
            clip_writer.release()
    
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

# Usage:
PAT_NOW = 'S20_150'
USE_BBOXES = True
SAVE_CLIPS = True
INPUT_FOLDER = f'/home/jgopal/NAS/Analysis/MP4/{PAT_NOW}_MP4'
CSV_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}'
VIDEO_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_Visualized/{PAT_NOW}/'
THRESHOLDS_CSV_PATH = f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/event_detection_counts_{PAT_NOW}.csv'
CLIPS_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/{PAT_NOW}/'

os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)

# Load the thresholds CSV
thresholds_df = pd.read_csv(THRESHOLDS_CSV_PATH)

# Function to find the highest threshold with at least 100 events
def find_highest_threshold(emotion):
    emotion_counts = thresholds_df[emotion]
    valid_thresholds = thresholds_df[emotion_counts >= 100]['Threshold']
    return valid_thresholds.max() if not valid_thresholds.empty else None

# Find the highest valid threshold for each emotion
happiness_threshold = find_highest_threshold('Happiness')
anger_threshold = find_highest_threshold('Anger')
neutral_threshold = find_highest_threshold('Neutral')
sadness_threshold = find_highest_threshold('Sadness')

# Create a DataFrame to store the chosen thresholds
thresholds_meta = pd.DataFrame({
    'Emotion': ['Happiness', 'Anger', 'Neutral', 'Sadness'],
    'Threshold': [happiness_threshold, anger_threshold, neutral_threshold, sadness_threshold]
})

# Define the path for the meta data CSV
meta_data_csv_path = os.path.join(CSV_OUTPUT_FOLDER, 'chosen_thresholds_meta.csv')

# Save the meta data CSV
thresholds_meta.to_csv(meta_data_csv_path, index=False)

print(f"Chosen thresholds meta data saved to: {meta_data_csv_path}")

# Visualize analysis for each emotion with the determined thresholds
if happiness_threshold:
    visualize_analysis(INPUT_FOLDER, CSV_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER, threshold=happiness_threshold, emotion_now='Happiness', USE_BBOXES=USE_BBOXES, SAVE_CLIPS=SAVE_CLIPS, CLIPS_OUTPUT_FOLDER=CLIPS_OUTPUT_FOLDER)

if anger_threshold:
    visualize_analysis(INPUT_FOLDER, CSV_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER, threshold=anger_threshold, emotion_now='Anger', USE_BBOXES=USE_BBOXES, SAVE_CLIPS=SAVE_CLIPS, CLIPS_OUTPUT_FOLDER=CLIPS_OUTPUT_FOLDER)

if neutral_threshold:
    visualize_analysis(INPUT_FOLDER, CSV_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER, threshold=neutral_threshold, emotion_now='Neutral', USE_BBOXES=USE_BBOXES, SAVE_CLIPS=SAVE_CLIPS, CLIPS_OUTPUT_FOLDER=CLIPS_OUTPUT_FOLDER)

if sadness_threshold:
    visualize_analysis(INPUT_FOLDER, CSV_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER, threshold=sadness_threshold, emotion_now='Sadness', USE_BBOXES=USE_BBOXES, SAVE_CLIPS=SAVE_CLIPS, CLIPS_OUTPUT_FOLDER=CLIPS_OUTPUT_FOLDER)
