python3 main_EventAnalysis_AllFrames.py 
 40%|██████████████████                           | 2/5 [00:13<00:19,  6.53s/it]
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 133, in <module>
    video_events = detect_events(emotion_df, au_df, openface_df)
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 75, in detect_events
    for frame in range(start_frame, end_frame + 1):
TypeError: 'numpy.float64' object cannot be interpreted as an integer
