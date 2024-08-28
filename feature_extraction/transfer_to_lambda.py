Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis.py", line 130, in <module>
    event_data = pd.merge(event_au_df, event_emotion_df.drop(columns=['frame', 'timestamp', 'success']), on='frame')
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/core/reshape/merge.py", line 170, in merge
    op = _MergeOperation(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/core/reshape/merge.py", line 794, in __init__
    ) = self._get_merge_keys()
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/core/reshape/merge.py", line 1297, in _get_merge_keys
    right_keys.append(right._get_label_or_level_values(rk))
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/core/generic.py", line 1911, in _get_label_or_level_values
    raise KeyError(key)
KeyError: 'frame'
