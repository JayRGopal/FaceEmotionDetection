 40%|██████████████████                           | 2/5 [00:13<00:19,  6.55s/it]
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 146, in <module>
    video_events = detect_events(emotion_df, au_df, openface_df)
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 55, in detect_events
    events[-1]['Duration in Seconds'] = round(events[-1]['Duration in Seconds'] + event_length / FACEDX_FPS, 1)
KeyError: 'Duration in Seconds'
