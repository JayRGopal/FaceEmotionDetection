import os
import pandas as pd
import cv2
import numpy as np

# Set the directories
PAT_NOW = 'S12'
VIDEO_DIRECTORY = os.path.abspath(f'/home/klab/NAS/OutpatientVideos/{PAT_NOW}/')
CSV_DIRECTORY = os.path.abspath(f'/home/klab/NAS/Analysis/outputs_Combined_Outpt/{PAT_NOW}/') 
CUT_VIDEO_FOLDER = os.path.abspath(f'/home/klab/NAS/OutpatientVideos_PatientOnly/{PAT_NOW}/')


def process_videos(video_dir, csv_dir, cut_video_folder):
    if not os.path.exists(cut_video_folder):
        os.makedirs(cut_video_folder)
    
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            csv_folder = os.path.join(csv_dir, video_file)
            csv_path = os.path.join(csv_folder, 'outputs_hse.csv')
            
            if os.path.exists(csv_path):
                out_path = os.path.join(cut_video_folder, 'patientOnly_' + video_file)
                if os.path.exists(out_path):
                    print(f'Skipping {out_path}: cut video already exists!')
                else:
                  print(f'Starting to cut video to save to {out_path}')
                  df = pd.read_csv(csv_path)
                  successful_frames = df[df['success'] == 1]['frame'].to_numpy()
                  frames_to_include = set()

                  #   for i in range(1, 5):
                  #       frames_to_include.update(successful_frames + i)
                  for frame in successful_frames:
                    next_frame = int(frame) + 5
                    if next_frame in successful_frames:
                        for i in range(int(frame), int(frame) + 6):
                            frames_to_include.update(i)

                  cap = cv2.VideoCapture(video_path)
                  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                  out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                  
                  frame_index = 0
                  while cap.isOpened():
                      ret, frame = cap.read()
                      if not ret:
                          break
                      if frame_index in frames_to_include:
                          out.write(frame)
                      frame_index += 1
                  
                  cap.release()
                  out.release()
                  
                  print(f"Saved cut video to {out_path}")
            else:
                print(f"No CSV found for {video_path}")

process_videos(VIDEO_DIRECTORY, CSV_DIRECTORY, CUT_VIDEO_FOLDER)
