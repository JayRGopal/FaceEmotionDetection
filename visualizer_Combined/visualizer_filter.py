import cv2
import pandas as pd
import numpy as np
import os

def visualize_analysis(input_folder, csv_output_folder, video_output_folder, threshold=0.8, USE_BBOXES=True):
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # Get the last part of the input folder path for naming the output video
    output_video_name = os.path.basename(os.path.normpath(input_folder)) + '_combined_output.mp4'
    combined_output_path = os.path.join(video_output_folder, output_video_name)
    
    # Initialize the VideoWriter object later
    out = None
    
    def draw_data_emotion(data, start_x, canvas):
        max_items_per_column = 8
        num_cols = ((len(list(data.columns)) - 3) + max_items_per_column - 1 ) // max_items_per_column
        col_width = width // num_cols
        for idx, (col, val) in enumerate(data.drop(columns=['frame', 'timestamp', 'success']).items()):
            color = (0, 0, 255) if val.values[0] > threshold else (0, 0, 0)
            text = f"{col}: {round(val.values[0], 3)}"
            x_offset = (idx // max_items_per_column) * col_width
            y_offset = 70 * (idx % max_items_per_column)  # Adjusted y_offset to accommodate bigger text
            cv2.putText(canvas, text, (start_x + x_offset, 70 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)  # Increased font scale and thickness
    
    def draw_data_au(data, start_x, canvas):
        max_items_per_column = 11
        num_cols = ((len(list(data.columns)) - 3) + max_items_per_column - 1 ) // max_items_per_column
        col_width = width // num_cols
        for idx, (col, val) in enumerate(data.drop(columns=['frame', 'timestamp', 'success']).items()):
            color = (0, 0, 255) if val.values[0] > threshold else (0, 0, 0)
            text = f"{col}: {round(val.values[0], 3)}"
            x_offset = (idx // max_items_per_column) * col_width
            y_offset = 60 * (idx % max_items_per_column)  # Adjusted y_offset to accommodate bigger text
            cv2.putText(canvas, text, (start_x + x_offset, 60 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)  # Increased font scale and thickness
    
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        output_base_path = os.path.splitext(video_file)[0]
        
        au_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_ogau.csv')
        emotion_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_hse.csv')
        bbox_csv_path = os.path.join(csv_output_folder, video_file, 'outputs_bboxes.csv')
        
        try:
            # Load CSVs
            au_df = pd.read_csv(au_csv_path)
            emotion_df = pd.read_csv(emotion_csv_path)
            if USE_BBOXES:
                bbox_df = pd.read_csv(bbox_csv_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping {video_file} due to empty CSV files.")
            continue
        
        # Filter frames based on emotion threshold
        filtered_frames = emotion_df[emotion_df['Happiness'] >= threshold]['frame'].unique()
        
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            if current_frame not in filtered_frames:
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
            
            # Extract facial bounding box coordinates for the current frame
            if USE_BBOXES and not bbox_data.empty:
                x, y, w, h = bbox_data.iloc[0]['Facial Box X'], bbox_data.iloc[0]['Facial Box Y'], bbox_data.iloc[0]['Facial Box W'], bbox_data.iloc[0]['Facial Box H']
                
                # Draw the facial bounding box on the frame
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Create a blank canvas
            canvas = 255 * np.ones((height, 3*width, 3), dtype=np.uint8)
            
            # Place the video frame in the middle
            canvas[:, width:2*width, :] = frame
            
            # Write the emotion outputs on the left
            if emotion_data is not None:
                draw_data_emotion(emotion_data, 10, canvas)
            
            # Write the AU outputs on the right
            if au_data is not None:
                draw_data_au(au_data, 2*width + 10, canvas)
            
            # Write the frame to the output video
            out.write(canvas)
        
        cap.release()
    
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

# Usage:
PAT_NOW = 'S20_150'
USE_BBOXES = False
INPUT_FOLDER = f'/home/jgopal/NAS/Analysis/MP4/{PAT_NOW}_MP4'
CSV_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}'
VIDEO_OUTPUT_FOLDER = f'/home/jgopal/NAS/Analysis/outputs_Visualized/{PAT_NOW}/'

os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)

visualize_analysis(INPUT_FOLDER, CSV_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER, threshold=0.8, USE_BBOXES=USE_BBOXES)