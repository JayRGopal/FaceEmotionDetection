 40%|██████████████████                           | 2/5 [00:13<00:19,  6.62s/it]
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 135, in <module>
    video_events = detect_events(emotion_df, au_df, openface_df)
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 91, in detect_events
    frame_data.update(frame_au.to_dict(orient='records')[0])
IndexError: list index out of range
