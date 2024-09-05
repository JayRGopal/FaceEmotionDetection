100%|█████████████████████████████████████████████| 5/5 [00:21<00:00,  4.38s/it]
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_EventAnalysis_AllFrames.py", line 160, in <module>
    events_df.to_csv(OUTPUT_CSV, index=False)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/util/_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/core/generic.py", line 3967, in to_csv
    return DataFrameRenderer(formatter).to_csv(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/io/formats/format.py", line 1014, in to_csv
    csv_formatter.save()
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/io/formats/csvs.py", line 251, in save
    with get_handle(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: '/home/jgopal/NAS/Analysis/outputs_EventAnalysis/combined_events_S23_199.csv'
