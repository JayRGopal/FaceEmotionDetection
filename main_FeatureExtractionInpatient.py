# This script does feature extraction AND linear regression for plotting.
# See main_FeatureExtractionInpatient_JustLinReg.py if feature extraction is complete, and you just need linear regression! 

PAT_NOW = "S23_199"
PAT_SHORT_NAME = "S_199"

print(f'[LOG] Patient Now: {PAT_NOW}')


MOOD_TRACKING_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'

BEHAVIORAL_LABELS_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Behavior_Labeling.xlsx'

VIDEO_TIMESTAMPS_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/videoDateTimes/VideoDatetimes{PAT_SHORT_NAME[1:]}.xlsx'

OPENFACE_OUTPUT_DIRECTORY = f'/home/jgopal/NAS/Analysis/outputs_OpenFace/{PAT_NOW}/'
COMBINED_OUTPUT_DIRECTORY = f'/home/jgopal/NAS/Analysis/outputs_Combined/{PAT_NOW}/'

RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/{PAT_SHORT_NAME}/'
FEATURE_VIS_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Feature_Visualization/{PAT_SHORT_NAME}/'
FEATURE_LABEL_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Feature_Labels/'
QC_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Quality_Control/'


EMO_FEATURE_SETTING = 2

# 0 - Our Custom AU --> Emotions, with all emotions
# 1 - Our Custom AU --> Emotions, with just OpenDBM's emotions
# 2 - OpenDBM's AU--> Emotions


STATS_FEATURE_SETTING = 3

# 0 - Our new features (including autocorrelation, kurtosis, etc.)
# 1 - Our new features, excluding extras like autocorrelation and kurtosis
# 2 - Just pres_pct
# 3 - Our new features, excluding extras. Do NOT threshold AUs before computing metrics. HSE gets 5 event features. OGAU gets num events and presence percent.

NORMALIZE_DATA = 0

# 0 - No time series normalization
# 1 - Yes time series normalization (for each time window)



import pandas as pd
import numpy as np
import os



import warnings
import pandas as pd

# Ignore all warnings
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')


# SAVE VARIABLES
import pickle


def get_var_name(our_variable):
    namespace = globals()
    for name, obj in namespace.items():
        if obj is our_variable:
            return name
    return None

# Save the dictionary to a file using pickle
def save_var(our_variable, RUNTIME_VAR_PATH=RUNTIME_VAR_PATH, forced_name=None):
    if forced_name is None:
        name_now = get_var_name(our_variable)
    else:
        name_now = forced_name

    # Construct the full path including the file name
    full_path = os.path.join(RUNTIME_VAR_PATH, f'{name_now}.pkl')

    # Ensure the directory exists, including any nested folders in name_now
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Save the variable
    with open(full_path, 'wb') as file:
        pickle.dump(our_variable, file)

def load_var(variable_name, RUNTIME_VAR_PATH=RUNTIME_VAR_PATH):
  # Load from the file
  with open(RUNTIME_VAR_PATH + f'{variable_name}.pkl', 'rb') as file:
      return pickle.load(file)


print('[LOG] Starter Functions Defined')
df = pd.read_excel(MOOD_TRACKING_SHEET_PATH, sheet_name=f'{PAT_SHORT_NAME}')

## Preprocess the mood tracking sheet

# Replace the P_number mood headers with just the mood
# df.columns = df.columns.str.replace('P[0-9]+ ', '')

# Properly deal with the missing values
df = df.replace('', np.nan).replace(' ', np.nan).fillna(value=np.nan)

df_moodTracking = df


df_moodTracking = df_moodTracking.drop(columns=['Notes'], errors='ignore')

df_moodTracking['Datetime'] = pd.to_datetime(df_moodTracking['Datetime']).dt.strftime('%-m/%-d/%Y %H:%M:%S')



import numpy as np

# create lists to hold the positive and negative affect items
pos_items = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
neg_items = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]

# get all columns that start with 'P' and split them into pos and neg groups
P_cols = [col for col in df_moodTracking.columns if col.startswith('P') and not(col.startswith('Pain')) and not(col.startswith('PANAS')) and not(col.startswith('Positive'))]
pos_cols = [col for col in P_cols if int(col[1:3]) in pos_items]
neg_cols = [col for col in P_cols if int(col[1:3]) in neg_items]

# create new columns for the summed scores
df_moodTracking['Positive Affect Score'] = df_moodTracking[pos_cols].fillna(0).astype(int).sum(axis=1, skipna=True)
df_moodTracking['Negative Affect Score'] = df_moodTracking[neg_cols].fillna(0).astype(int).sum(axis=1, skipna=True)
df_moodTracking['Overall Affect Score'] = df_moodTracking[['Positive Affect Score', 'Negative Affect Score']].fillna(0).astype(int).sum(axis=1, skipna=True)

# replace 0s with NaNs in columns 'Positive Affect Score' and 'Negative Affect Score'
df_moodTracking[['Positive Affect Score', 'Negative Affect Score', 'Overall Affect Score']] = \
            df_moodTracking[['Positive Affect Score', 'Negative Affect Score', 'Overall Affect Score']].replace(0, np.nan)

# drop the original P columns used to create the scores
df_moodTracking.drop(columns=pos_cols + neg_cols, inplace=True)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def normalize_columns(df, method=1):
    # Create a copy of the DataFrame
    normalized_df = df.copy()

    # Get the column names excluding 'Datetime'
    columns_to_normalize = [col for col in normalized_df.columns if col != 'Datetime']

    if method == 1:
        # No scaling or normalization
        pass

    elif method == 2:
        # MinMax scaling to range 0 to 10
        scaler = MinMaxScaler(feature_range=(0, 10))
        normalized_df[columns_to_normalize] = scaler.fit_transform(normalized_df[columns_to_normalize])

    elif method == 3:
        # MinMax scaling to range 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_df[columns_to_normalize] = scaler.fit_transform(normalized_df[columns_to_normalize])

    elif method == 4:
        # Log scaling
        normalized_df[columns_to_normalize] = normalized_df[columns_to_normalize].astype(float)
        normalized_df[columns_to_normalize] = np.log1p(normalized_df[columns_to_normalize])

    elif method == 5:
        # Standard normalization (Z-score normalization)
        scaler = StandardScaler()
        normalized_df[columns_to_normalize] = scaler.fit_transform(normalized_df[columns_to_normalize])

    else:
        raise ValueError("Invalid method. Choose a value between 1 and 5.")

    return normalized_df


df_moodTracking = normalize_columns(df_moodTracking, method=2)

if PAT_SHORT_NAME == 'S_214':
    df_moodTracking = df_moodTracking.drop(1).reset_index(drop=True)

df_videoTimestamps = pd.read_excel(VIDEO_TIMESTAMPS_SHEET_PATH, sheet_name=f'VideoDatetimes_{PAT_SHORT_NAME.split("_")[-1]}')
df_videoTimestamps['Filename'] = df_videoTimestamps['Filename'].str.replace('.m2t', '')

if PAT_SHORT_NAME == 'S_199':
  # There's no H01 video, so let's drop that filename
  df_videoTimestamps = df_videoTimestamps.drop(211)

print('[LOG] Labels Processed')


# Check for any missing videos!

def print_difference(list1, list2):
    for item in list1:
        if item not in list2:
            print(item)

filenames_master_list = list(df_videoTimestamps['Filename'].values)
filenames_we_have = [i[:-4] for i in os.listdir(COMBINED_OUTPUT_DIRECTORY)]

print_difference(filenames_master_list, filenames_we_have)


# DICTIONARY OF SEPARATE DFS

def get_dict_openface(output_dir):
    # Create an empty dictionary to hold the DataFrames
    dfs_openface = {}

    # Get a list of all the CSV files in the directory
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])

    # List of columns to keep
    columns_to_keep = [
        'frame', 'timestamp', 'success',
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 
        'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 
        'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU45_c'
    ]

    failed_files = []

    # Loop through the CSV files
    for csv_file in csv_files:
        try:
            # Load data into a pandas DataFrame
            csv_file_path = os.path.join(output_dir, csv_file)
            df_temp = pd.read_csv(csv_file_path)
            df_temp.columns = df_temp.columns.str.strip()

            # Keep every 6th row so it's 5 fps!
            X = 6
            df_temp = df_temp[df_temp.index % X == 0]

            # Filter DataFrame to keep only columns in list
            df_temp = df_temp.loc[:, columns_to_keep]

            # Fix column names to not have leading or trailing spaces
            df_temp = df_temp.rename(columns=lambda x: x.strip())

            # Store the DataFrame in the dictionary with the csv file name as the key
            # Remove the '.csv' by doing csv_file[:-4]
            dfs_openface[csv_file[:-4]] = df_temp

        except Exception as e:
            print(f"Failed to load {csv_file}: {str(e)}")
            failed_files.append(csv_file)

    if failed_files:
        raise Exception(f"Errors occurred while processing the following files: {', '.join(failed_files)}")

    return dfs_openface


def get_dict_openface_extras(output_dir):
  # Create an empty dictionary to hold the DataFrames
  dfs_openface = {}

  # Get a list of all the CSV files in the directory
  csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])

  # list of columns to keep
  columns_to_keep = ['frame', ' timestamp', ' success',
                     'gaze_0_x',
                     'gaze_0_y',
                     'gaze_0_z',
                     'gaze_1_x',
                     'gaze_1_y',
                     'gaze_1_z',
                     'pose_Tx',
                     'pose_Ty',
                     'pose_Tz',
                     'pose_Rx',
                     'pose_Ry',
                     'pose_Rz']

  columns_to_keep = columns_to_keep + [f"eye_lmk_X_{i}" for i in range(56)] + [f"eye_lmk_Y_{i}" for i in range(56)] + [f"eye_lmk_Z_{i}" for i in range(56)] 
  columns_to_keep = columns_to_keep + [f"X_{i}" for i in range(68)] + [f"Y_{i}" for i in range(68)] + [f"Z_{i}" for i in range(68)]
    
  # remove special character 
  columns_to_keep = [one_str.replace(' ', '') for one_str in columns_to_keep]

  # Loop through the CSV files
  for csv_file in csv_files:
      # Load data into a pandas df
      csv_file_path = os.path.join(output_dir, csv_file)
      df_temp = pd.read_csv(csv_file_path)
      df_temp.columns = df_temp.columns.str.strip()

      # keep every 6th row such that it's 5 fps!
      X = 6
      df_temp = df_temp[df_temp.index % X == 0]

      # filter DataFrame to keep only columns in list
      # remove special character 
      df_temp = df_temp.loc[:, columns_to_keep]

      # fix column names to not have leading or trailing spaces!
      df_temp = df_temp.rename(columns=lambda x: x.strip())

      # Store the DataFrame in the dictionary with the csv file name as the key
      # remove the '.csv' by doing csv_file[:-4]
      dfs_openface[csv_file[:-4]] = df_temp
      del df_temp

  return dfs_openface



def only_successful_frames(df):
    # get frames where AU/emotion detection was successful!
    return df[df['success'] == 1]

def apply_function_to_dict(dictionary, func, **kwargs):
    """
    Apply a function to each DataFrame in a dictionary and return a modified copy of the dictionary.

    Args:
        dictionary (dict): The dictionary containing DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict: A modified copy of the dictionary with the function applied to each DataFrame.
    """
    return {key: func(df, **kwargs) for key, df in dictionary.items()}



print('[LOG] Loading in OpenFace Outputs')

# Check and load or generate dfs_openface
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_openface_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_openface if not already saved
    dfs_openface = get_dict_openface(OPENFACE_OUTPUT_DIRECTORY)
    dfs_openface = apply_function_to_dict(dfs_openface, only_successful_frames)
    save_var(dfs_openface, forced_name=f'dfs_openface_{PAT_SHORT_NAME}')
else:
    # Load dfs_openface if it already exists
    dfs_openface = load_var(f'dfs_openface_{PAT_SHORT_NAME}')



# Check and load or generate dfs_openface_extras
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_openface_extras_{PAT_NOW}.pkl'):
    # Generate dfs_openface_extras if not already saved
    dfs_openface_extras = get_dict_openface_extras(OPENFACE_OUTPUT_DIRECTORY)
    dfs_openface_extras = apply_function_to_dict(dfs_openface_extras, only_successful_frames)
    save_var(dfs_openface_extras, forced_name=f'dfs_openface_extras_{PAT_NOW}')
else:
    # Load dfs_openface_extras if it already exists
    dfs_openface_extras = load_var(f'dfs_openface_extras_{PAT_NOW}')

print('[LOG] OpenFace Outputs Loaded In')




import pandas as pd
import os

def get_dict(output_dir, file_now='outputs_hse.csv', filterOutLR=True):

  # Initialize an empty dictionary to store the dataframes
  df_dict = {}

  # Loop through the subfolders in alphabetical order
  for subfolder_name in sorted(os.listdir(output_dir)):

    # Check if the subfolder contains CSV files
    subfolder_path = os.path.join(output_dir, subfolder_name)
    if not os.path.isdir(subfolder_path):
      continue

    # Load the first CSV file in the subfolder into a dataframe
    csv_file_path = os.path.join(subfolder_path, file_now)
    if not os.path.isfile(csv_file_path):
      continue

    try:
      df_temp = pd.read_csv(csv_file_path)
    except:
      df_temp = pd.DataFrame(columns=['frame', 'timestamp', 'success', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9',
       'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18',
       'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32',
       'AU38', 'AU39'])


    # OpenGraphAU - we are filtering out L and R!
    if filterOutLR:
      df_temp = df_temp.filter(regex='^(?!AUL|AUR)')

    # Add the dataframe to the dictionary with the subfolder name as the key
    # We do [:-4] to remove '.mp4' from the end of the string
    df_dict[subfolder_name[:-4]] = df_temp

  return df_dict

def create_binary_columns(df, threshold):
    df_copy = df.copy()
    # adds classification columns to opengraphAU
    for col in df_copy.columns:
        if col.startswith('AU'):
            # Add _c to the column name for the new column
            new_col_name = col + '_c'
            # Apply the binary classification to the new column
            df_copy[new_col_name] = df_copy[col].apply(lambda x: 1 if x >= threshold else 0)
            # Add _r to the original column name
            df_copy = df_copy.rename(columns={col: col + '_r'}, inplace=False)
    return df_copy

def remove_columns_ending_with_r(df):
    columns_to_drop = [col for col in df.columns if col.endswith('_r')]
    df = df.drop(columns=columns_to_drop, inplace=False)
    return df


def only_successful_frames(df):
    # get frames where AU/emotion detection was successful!
    return df[df['success'] == 1]


def apply_function_to_dict(dictionary, func, **kwargs):
    """
    Apply a function to each DataFrame in a dictionary and return a modified copy of the dictionary.

    Args:
        dictionary (dict): The dictionary containing DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict: A modified copy of the dictionary with the function applied to each DataFrame.
    """
    return {key: func(df, **kwargs) for key, df in dictionary.items()}

print('[LOG] Loading in HSE Outputs')
# Check and load or generate dfs_hsemotion
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_hsemotion_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_hsemotion if not already saved
    dfs_hsemotion = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_hse.csv')
    dfs_hsemotion = apply_function_to_dict(dfs_hsemotion, only_successful_frames)
    save_var(dfs_hsemotion, forced_name=f'dfs_hsemotion_{PAT_SHORT_NAME}')
else:
    # Load dfs_hsemotion if it already exists
    dfs_hsemotion = load_var(f'dfs_hsemotion_{PAT_SHORT_NAME}')

print('[LOG] HSE Outputs Loaded In')

print('[LOG] Loading in OGAU Outputs')

# Check and load or generate dfs_opengraphau
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_opengraphau_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_opengraphau if not already saved
    OPENGRAPHAU_THRESHOLD = 0.5
    dfs_opengraphau = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_ogau.csv')
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, create_binary_columns, threshold=OPENGRAPHAU_THRESHOLD)
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, only_successful_frames)
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, remove_columns_ending_with_r)
    save_var(dfs_opengraphau, forced_name=f'dfs_opengraphau_{PAT_SHORT_NAME}')
else:
    # Load dfs_opengraphau if it already exists
    dfs_opengraphau = load_var(f'dfs_opengraphau_{PAT_SHORT_NAME}')

print('[LOG] OGAU Outputs Loaded In')


def get_data_within_duration(dfs_dict, df_video_timestamps, datetime, duration):
    # Takes in:
    # dfs_dict -- a dictionary of dataframes containing csv data from one of the pipelines
    # df_video_timestamps -- the VideoDateTimes_199 csv
    # datetime -- a pd.datetime value to center our extraction
    # duration -- a duration (in minutes) BEFORE the datetime to extract

    # Outputs:
    # One dataframe with all rows we want, with timestamps converted into correct datetimes
    start_datetime = datetime - pd.Timedelta(minutes=duration)
    end_datetime = datetime

    relevant_keys = df_video_timestamps.loc[(pd.to_datetime(df_video_timestamps['VideoEnd']) >= start_datetime) &
                                            (pd.to_datetime(df_video_timestamps['VideoStart']) <= end_datetime), 'Filename'].values

    relevant_dfs = []
    for key in relevant_keys:
        if key in dfs_dict:
            video_start = pd.to_datetime(df_video_timestamps.loc[df_video_timestamps['Filename'] == key, 'VideoStart'].values[0])
            video_end = pd.to_datetime(df_video_timestamps.loc[df_video_timestamps['Filename'] == key, 'VideoEnd'].values[0])
            time_mask = ((dfs_dict[key]['timestamp'] >= (start_datetime - video_start).total_seconds()) &
                         (dfs_dict[key]['timestamp'] <= (end_datetime - video_start).total_seconds()))
            df = dfs_dict[key].loc[time_mask].copy()
            df['timestamp'] = video_start + pd.to_timedelta(df['timestamp'], unit='s')
            relevant_dfs.append(df)

    if relevant_dfs:
        df_combined = pd.concat(relevant_dfs, ignore_index=True, sort=False)
        df_combined = df_combined.drop(columns='frame')

        return df_combined

    print(f"MAJOR ERROR! ZERO RELEVANT DFS!! DATETIME: {datetime}")
    return pd.DataFrame()

def get_radius_dict(TIME_RADIUS_IN_MINUTES, INPUT_DF, df_videoTimestamps, df_moodTracking, takeAll=True):
  # takes in the:
  # --time radius,
  # --input dataframe dict (e.g. is it from OpenFace? HSEmotion?)
  # --df with video timestamps
  # --df with mood tracking patient reports
  # --takeAll - are we taking all reports, or filtering out values w/o mood (e.g. anxiety)? True = no filtering

  # returns dictionary of timestamp : df with relevant frames

  # We'll make a dictionary, with the relevant df for each datetime we have a report
  radius_df_dict = {}
  for oneIndex in range(len(df_moodTracking)):
    # Let's make sure there's a value collected (or takeAll = True)!
    if takeAll:
      dt_now = get_moodTracking_datetime(oneIndex, df_moodTracking=df_moodTracking)
      filtered_df = get_data_within_duration(INPUT_DF, df_videoTimestamps, dt_now, TIME_RADIUS_IN_MINUTES)
      radius_df_dict[dt_now] = filtered_df
    else:
      val_now = df_moodTracking[oneIndex:oneIndex+1]['Anxiety'][oneIndex]
      if isinstance(val_now, str):
        # Value was collected
        dt_now = get_moodTracking_datetime(oneIndex, df_moodTracking=df_moodTracking)
        filtered_df = get_data_within_duration(INPUT_DF, df_videoTimestamps, dt_now, TIME_RADIUS_IN_MINUTES)
        radius_df_dict[dt_now] = filtered_df
      else:
        # No value collected!
        print('No value for Anxiety for index ', oneIndex, f'corresponding to {get_moodTracking_datetime(oneIndex, df_moodTracking=df_moodTracking)}')
  return radius_df_dict

def generate_number_list(start, interval, count):
    number_list = [start + i * interval for i in range(count)]
    return number_list

def get_moodTracking_datetime(index, df_moodTracking):
  temp_var = pd.to_datetime(pd.to_datetime(df_moodTracking[index:index+1]['Datetime']).dt.strftime('%d-%b-%Y %H:%M:%S'))
  return pd.Timestamp(temp_var[index])


# EMOTION DETECTION & AFFECT

takeAll = True # we are taking all patient reports

# start and interval are in minutes
TIME_RADIUS_LIST = generate_number_list(start=15, interval=15, count=16)
#TIME_RADIUS_LIST = [60, 120, 180, 240]




ENABLE_OPENFACE = True

if ENABLE_OPENFACE:
  openface_radius_dict = {}
  openface_extras_radius_dict = {}
  

hsemotion_radius_dict = {}
opengraphau_radius_dict = {}

print('[LOG] Creating Time Radius Dicts')

for i in TIME_RADIUS_LIST:
  if ENABLE_OPENFACE:
    openface_radius_now = get_radius_dict(i, dfs_openface, df_videoTimestamps, df_moodTracking, takeAll=takeAll)
    openface_radius_dict[f'{i}'] = openface_radius_now
    
    openface_extras_radius_now = get_radius_dict(i, dfs_openface_extras, df_videoTimestamps, df_moodTracking, takeAll=takeAll)
    openface_extras_radius_dict[f'{i}'] = openface_extras_radius_now

  hsemotion_radius_now = get_radius_dict(i, dfs_hsemotion, df_videoTimestamps, df_moodTracking, takeAll=takeAll)
  hsemotion_radius_dict[f'{i}'] = hsemotion_radius_now

  opengraphau_radius_now = get_radius_dict(i, dfs_opengraphau, df_videoTimestamps, df_moodTracking, takeAll=takeAll)
  opengraphau_radius_dict[f'{i}'] = opengraphau_radius_now

print('[LOG] Time Radius Dicts Created')

def time_splitter(input_dict, splitter_times):
    # Initialize the output dictionary
    output_dict = {}
    
    # Frame rate: 5 frames per second
    frame_rate = 5
    
    # Iterate over each split time
    for split_time in splitter_times:
        # Initialize the dictionary for the current split time
        output_dict[split_time] = {}
        
        # Calculate the number of rows per split
        rows_per_split = split_time * 60 * frame_rate
        
        # Iterate over the outer dictionary
        for outer_key, inner_dict in input_dict.items():
            # Initialize the dictionary for the current outer key
            output_dict[split_time][outer_key] = {}
            
            # Iterate over the inner dictionary
            for timestamp, df in inner_dict.items():
                # Split the DataFrame into chunks of the specified size
                split_dfs = [df.iloc[i:i+rows_per_split] for i in range(0, len(df), rows_per_split)]
                
                # Assign the list of split DataFrames to the appropriate location in the output dictionary
                output_dict[split_time][outer_key][timestamp] = split_dfs
    
    return output_dict

# Functions to apply feature processing to the inpatient dictionary structure

def apply_function_to_dict_list(dictionary, func, **kwargs):
    """
    Apply a function to each DataFrame in a dictionary where values are LISTS of dfs and return a modified copy of the dictionary.

    Args:
        dictionary (dict): The dictionary containing DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict: A modified copy of the dictionary with the function applied to each DataFrame.
    """
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                new_dict[split_time][outer_key][timestamp] = [func(df, **kwargs) for df in df_list]
    return new_dict

def average_inner_dfs(dictionary):
    """
    Replace each list of DataFrames in a nested dictionary with the average of the DataFrames in each list.
    For columns with strings, convert to numbers if possible and take the string from the first DataFrame otherwise.

    Args:
        dictionary (dict): The dictionary containing lists of DataFrames.

    Returns:
        dict: A modified copy of the dictionary with the average of each list of DataFrames.
    """
    def process_columns(df_list):
        """
        Process columns to calculate averages for numeric columns and keep strings from the first DataFrame.
        """
        if not df_list:
            # If df_list is empty, return an empty DataFrame
            return pd.DataFrame()

        combined_df = pd.concat(df_list, ignore_index=True)
        avg_df = pd.DataFrame(index=combined_df.index)

        for column in combined_df.columns:
            # Try to convert the column to numeric
            numeric_series = pd.to_numeric(combined_df[column], errors='coerce')
            
            if numeric_series.notna().all():
                # If all values can be converted to numeric, calculate the mean
                avg_df[column] = numeric_series.groupby(combined_df.index).mean()
            else:
                # Repeat the first DataFrame's values to match the length of the combined DataFrame
                avg_df[column] = np.tile(df_list[0][column].values[0], len(combined_df))
        
        return avg_df
    
    def create_empty_df_like(sample_df):
        """
        Create a DataFrame with the same columns as sample_df but filled with zeros (or equivalent) based on datatype.
        """
        return pd.DataFrame({col: np.zeros(sample_df.shape[0], dtype=sample_df[col].dtype) for col in sample_df.columns})

    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                if df_list:
                    # If the df_list is not empty, process normally
                    avg_df = process_columns(df_list)
                else:
                    # If the df_list is empty, find a non-empty DataFrame structure to create a zero-filled DataFrame
                    for outer_split_time in dictionary.values():
                        for outer_inner_dict in outer_split_time.values():
                            for df in outer_inner_dict.values():
                                if df:  # Ensure df_list is not empty
                                    avg_df = create_empty_df_like(df[0])
                                    break
                            if 'avg_df' in locals():
                                break
                        if 'avg_df' in locals():
                            break
                new_dict[split_time][outer_key][timestamp] = avg_df
    return new_dict


print('[LOG] Applying 5 Min Time Split to Radius Dicts')

openface_radius_dict = time_splitter(openface_radius_dict, [5, 10])
save_var(openface_radius_dict, forced_name=f'openface_radius_dict_{PAT_SHORT_NAME}')
hsemotion_radius_dict = time_splitter(hsemotion_radius_dict, [5, 10])
save_var(hsemotion_radius_dict, forced_name=f'hsemotion_radius_dict_{PAT_SHORT_NAME}')
opengraphau_radius_dict = time_splitter(opengraphau_radius_dict, [5, 10])
save_var(opengraphau_radius_dict, forced_name=f'opengraphau_radius_dict_{PAT_SHORT_NAME}')
openface_extras_radius_dict = time_splitter(openface_extras_radius_dict, [5, 10])
save_var(openface_extras_radius_dict, forced_name=f'openface_extras_radius_dict_{PAT_SHORT_NAME}')


print('[LOG] 5 Min Time Splitter Applied to Radius Dicts')



print('[LOG] Beginning Feature Extraction')
# Define emotion to AU mapping

# OpenDBM:
emo_AUs = {'Happiness': [6, 12],
           'Sadness': [1, 4, 15],
           'Surprise': [1, 2, 5, 26],
           'Fear': [1, 2, 4, 5, 7, 20, 26],
           'Anger': [4, 5, 7, 23],
           'Disgust': [9, 15, 16],
           'Contempt': [12, 14]}


# Define AU to lower/upper

# OpenDBM:
AU_lower = [12, 15, 26, 20, 23, 14]
AU_upper = [6, 1, 4, 2, 5, 7, 9]


def only_successful_frames(df):
    # get frames where AU/emotion detection was successful!
    return df[df['success'] == 1]

from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf

def binarize_cols(df, threshold=0.5):
  new_df = df.copy()
  emotions = [col for col in new_df.columns if col not in ['frame', 'success', 'timestamp']]

  for emotion in emotions:
      new_df[f'{emotion}_Raw'] = new_df[emotion]
      new_df[f'{emotion}_Binary'] = (new_df[f'{emotion}_Raw'] >= threshold).astype(int)

  new_df = new_df.drop(columns=emotions, inplace=False)

  return new_df


def fill_empty_dfs_lists(dictionary):
    """
    Fill empty DataFrames in a nested dictionary structure with a DataFrame of zeros.
    
    Args:
        dictionary (dict): The dictionary containing nested dictionaries with lists of DataFrames.
    
    Returns:
        dict: A modified copy of the dictionary with empty DataFrames filled with zeros.
    """
    # Find the first non-empty DataFrame to use as a template for filling empty DataFrames
    non_empty_df = None
    for split_time, outer_dict in dictionary.items():
        for outer_key, inner_dict in outer_dict.items():
            for timestamp, df_list in inner_dict.items():
                for df in df_list:
                    if not df.empty:
                        non_empty_df = df
                        break
                if non_empty_df is not None:
                    break
            if non_empty_df is not None:
                break
        if non_empty_df is not None:
            break
    
    # If no non-empty DataFrame is found, return the original dictionary
    if non_empty_df is None:
        return dictionary

    # Create the modified dictionary
    modified_dictionary = {}
    for split_time, outer_dict in dictionary.items():
        modified_dictionary[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            modified_dictionary[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                modified_df_list = []
                for df in df_list:
                    if df.empty:
                        modified_df = pd.DataFrame(0, index=non_empty_df.index, columns=non_empty_df.columns)
                        # Preserve string columns from the non-empty DataFrame
                        for column in non_empty_df.columns:
                            if non_empty_df[column].dtype == object:
                                modified_df[column] = non_empty_df[column]
                    else:
                        modified_df = df.copy()
                    modified_df_list.append(modified_df)
                modified_dictionary[split_time][outer_key][timestamp] = modified_df_list

    return modified_dictionary


def analyze_emotion_events_v2(df, max_frame_gap=10, event_minimum_num_frames=1, method='HSE'):
    df = df.reset_index(drop=True)

    # Emotions to analyze
    emotions_raw = [col for col in df.columns if col not in ['frame', 'success', 'timestamp']]
    # Removing "_Raw" or "_Binary" from each string
    processed_strings = [s.replace("_Raw", "").replace("_Binary", "") for s in emotions_raw]
    # Eliminating duplicates
    emotions = list(set(processed_strings))

    # Create DataFrame for results
    if STATS_FEATURE_SETTING == 0:
        results_df = pd.DataFrame(index=['avg_event_length', 'avg_event_duration', 'total_num_events', 'avg_probability', 'std', 'skewness', 'kurtosis', 'autocorrelation', 'pres_pct'])
    elif STATS_FEATURE_SETTING == 1 or (STATS_FEATURE_SETTING == 3 and method == 'HSE'):
        results_df = pd.DataFrame(index=['avg_event_length', 'total_num_events', 'avg_probability', 'std', 'pres_pct'])
    elif STATS_FEATURE_SETTING == 2:
        results_df = pd.DataFrame(index=['pres_pct'])
    elif STATS_FEATURE_SETTING == 3 and (method == 'OGAU' or method=='OF'):
        results_df = pd.DataFrame(index=['pres_pct', 'total_num_events'])


    def detect_events(emotion_binary_col):
        probThreshold = 0.5 # irrelevant because it's a binary column
        minInterval = max_frame_gap
        minDuration = event_minimum_num_frames

        probBinary = emotion_binary_col > probThreshold

        # Using np.diff to find changes in the binary array
        changes = np.diff(probBinary.astype(int))

        # Identify start (1) and stop (-1) points
        starts = np.where(changes == 1)[0] + 1  # +1 to correct the index shift caused by diff
        stops = np.where(changes == -1)[0] + 1

        # Adjust for edge cases
        if probBinary.iloc[0]:
            starts = np.insert(starts, 0, 0)
        if probBinary.iloc[-1]:
            stops = np.append(stops, len(probBinary))

        # Merge close events and filter by duration
        events = []
        for start, stop in zip(starts, stops):

            # Construct the event considering only indices where probBinary is 1
            event = np.arange(start, stop)[probBinary[start:stop].values]

            # Check if there is a previous event to potentially merge with
            if events and event.size > 0 and events[-1][-1] >= start - minInterval:
                # Merge with the previous event
                events[-1] = np.unique(np.concatenate([events[-1], event]))
            elif event.size >= event_minimum_num_frames:
                events.append(event)

        # Filter events by minimum duration
        valid_events = [event for event in events if len(event) >= minDuration]

        return valid_events

    for emotion in emotions:
        # Identify events
        emotion_binary_col = df[f'{emotion}_Binary']
        emotion_presence = df[f'{emotion}_Binary'].sum()
        pres_pct = emotion_presence / len(df) * 100  # Percentage of frames where emotion is present
        events = detect_events(emotion_binary_col)

        if not(STATS_FEATURE_SETTING == 2):
            # Calculate features for each event
            if events:
                event_lengths = [len(event) for event in events]
                event_durations = [event[-1] - event[0] + 1 for event in events]
                probabilities = [df.loc[event, f'{emotion}_Raw'].values for event in events]
                probabilities_flattened = np.concatenate(probabilities)

                avg_event_length = np.mean(event_lengths)
                avg_event_duration = np.mean(event_durations)

                total_num_events = len(events)

                # NORMALIZE TOTAL NUM EVENTS BASED ON DF SIZE
                # total_num_events = len(events) * 1000 / df.shape[0]

                avg_probability = np.mean(probabilities_flattened)
                std_dev = np.std(probabilities_flattened)
                skewness_val = skew(probabilities_flattened)
                kurtosis_val = kurtosis(probabilities_flattened)
                autocorr = acf(probabilities_flattened, fft=True, nlags=1)[1] if len(probabilities_flattened) > 1 else 0
            else:
                avg_event_length = 0
                avg_event_duration = 0
                total_num_events = 0
                avg_probability = 0
                std_dev = 0
                skewness_val = 0
                kurtosis_val = 0
                autocorr = 0

        # Add results to the DataFrame
        if STATS_FEATURE_SETTING == 0:
            results_df[emotion] = [avg_event_length, avg_event_duration, total_num_events, avg_probability, std_dev, skewness_val, kurtosis_val, autocorr, pres_pct]
        elif STATS_FEATURE_SETTING == 1 or (STATS_FEATURE_SETTING == 3 and method == 'HSE'):
            results_df[emotion] = [avg_event_length, total_num_events, avg_probability, std_dev, pres_pct]
        elif STATS_FEATURE_SETTING == 2:
            results_df[emotion] = [pres_pct]
        elif STATS_FEATURE_SETTING == 3 and (method == 'OGAU' or method=='OF'):
            results_df[emotion] = [pres_pct, total_num_events]

    # Replace NaN values with 0
    results_df.fillna(0, inplace=True)

    return results_df



import scipy.stats as stats

def detect_emotions(df, method, emo_AUs, additional_filter=None):
    # INPUT:
    # df -- dataframe with AUs for each frame
    # method -- must be 'OpenFace'
    # emo_AUs -- the hash table
    # additional_filter -- are we just doing lower half? upper half? This is None or a list of ints (which AUs to keep)

    # OUTPUT:
    # 3 datafrmes. Each has emotion values for each frame
    # emo_hard, emo_soft, emo_binary (see OpenDBM docs for details)


    if df.empty:
      return (df, df, df)
    # We start by mapping AUs to emotions for each of our two methods
    # Using this mapping: https://aicure.github.io/open_dbm/docs/emotional-expressivity
    if method == 'OpenFace':
        columns = ['AU01_r','AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r',
                    'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
                    'AU26_r', 'AU45_r',
                    'AU01_c',
                    'AU02_c',
                    'AU04_c',
                    'AU05_c',
                    'AU06_c',
                    'AU07_c',
                    'AU09_c',
                    'AU10_c',
                    'AU12_c',
                    'AU14_c',
                    'AU15_c',
                    'AU17_c',
                    'AU20_c',
                    'AU23_c',
                    'AU25_c',
                    'AU26_c',
                    'AU45_c']

        # hash tables for presence and intensity
        emo_AUs_presence = {}
        emo_AUs_intensity = {}
        for key in emo_AUs.keys(): # loop through emotion strings
            new_values_r = [] # regression
            new_values_c = [] # classification

            for value in emo_AUs[key]:
                if isinstance(value, int):
                    AU_key_r = "AU{:02d}_r".format(value)
                    AU_key_c = "AU{:02d}_c".format(value)
                    if AU_key_r in columns:
                        if additional_filter is not None:
                          if value in additional_filter:
                            new_values_r.append(AU_key_r)
                        else:
                          new_values_r.append(AU_key_r)
                    if AU_key_c in columns:
                        if additional_filter is not None:
                          if value in additional_filter:
                            new_values_c.append(AU_key_c)
                        else:
                          new_values_c.append(AU_key_c)
            if new_values_r:
                emo_AUs_intensity[key] = new_values_r
            if new_values_c:
                emo_AUs_presence[key] = new_values_c

    else:
        # if the method specified is not OpenFace or OpenGraphAU, raise an error (pipeline doesn't support others yet)
        raise ValueError("Invalid method parameter. Method must be 'OpenFace'.")

    # Create an empty dictionary to store the emotion scores
    emotion_scores_hard = {} # only non-zero if all AUs present
    emotion_scores_soft = {} # average of AU intensities even if all not present
    emotion_scores_binary = {} # 1 or 0: are all AUs present?

    # Compute emotion scores for each emotion
    for emotion in emo_AUs_presence.keys():
        # Get the relevant columns for presence and intensity
        presence_cols = emo_AUs_presence[emotion]
        intensity_cols = emo_AUs_intensity[emotion]

        # Compute the emotion score for each row in the dataframe
        emotion_scores_hard[emotion] = df[intensity_cols].mean(axis=1) * df[presence_cols].all(axis=1)
        emotion_scores_hard[emotion] = emotion_scores_hard[emotion].fillna(0)

        emotion_scores_soft[emotion] = df[intensity_cols].mean(axis=1)
        emotion_scores_soft[emotion] = emotion_scores_soft[emotion].fillna(0)

        emotion_scores_binary[emotion] = df[presence_cols].all(axis=1)
        emotion_scores_binary[emotion] = emotion_scores_binary[emotion].fillna(0)

    # Create a new dataframe with the emotion scores
    emotion_df_hard = pd.DataFrame(emotion_scores_hard)
    emotion_df_soft = pd.DataFrame(emotion_scores_soft)
    emotion_df_binary = pd.DataFrame(emotion_scores_binary)
    emotion_df_binary = emotion_df_binary.replace({False: 0, True: 1})

    # Let's add timestamp and success on
    columns_of_interest = ['timestamp', 'success']
    df_temp = df[columns_of_interest]

    # Concatenate the columns from df2 with df1
    emotion_df_hard = pd.concat([df_temp, emotion_df_hard], axis=1)
    emotion_df_soft = pd.concat([df_temp, emotion_df_soft], axis=1)
    emotion_df_binary = pd.concat([df_temp, emotion_df_binary], axis=1)

    return emotion_df_hard, emotion_df_soft, emotion_df_binary



def detect_emotions_og(df, method, emo_AUs, additional_filter=None):
    # INPUT:
    # df -- dataframe with AUs for each frame
    # method -- must be 'OpenGraphAU'
    # emo_AUs -- the hash table
    # additional_filter -- are we just doing lower half? upper half? This is None or a list of ints (which AUs to keep)

    # OUTPUT:
    # 1 datafrme with emotion values for each frame
    # emo_binary (see OpenDBM docs for details)


    if df.empty:
      return df
    # We start by mapping AUs to emotions for each of our two methods
    # Using this mapping: https://aicure.github.io/open_dbm/docs/emotional-expressivity


    if method == 'OpenGraphAU':
        columns = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9',
                   'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17',
                   'AU18', 'AU19', 'AU20', 'AU22', 'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32',
                   'AU38', 'AU39']

        # add the classification columns!
        columns = [item for sublist in [[col+'_r', col+'_c'] for col in columns] for item in sublist]

        # hash tables for presence and intensity
        emo_AUs_presence = {}
        for key in emo_AUs.keys():
            new_values_c = []
            for value in emo_AUs[key]:
                if isinstance(value, int):
                    AU_key_c = f"AU{value}_c"

                    if AU_key_c in columns:
                        if additional_filter is not None:
                          if value in additional_filter:
                            new_values_c.append(AU_key_c)
                        else:
                          new_values_c.append(AU_key_c)
            if new_values_c:
                emo_AUs_presence[key] = new_values_c

    else:
        # if the method specified is not OpenFace or OpenGraphAU, raise an error (pipeline doesn't support others yet)
        raise ValueError("Invalid method parameter. Method must be 'OpenGraphAU'.")

    # Create an empty dictionary to store the emotion scores
    emotion_scores_binary = {} # 1 or 0: are all AUs present?

    # Compute emotion scores for each emotion
    for emotion in emo_AUs_presence.keys():
        # Get the relevant columns for presence
        presence_cols = emo_AUs_presence[emotion]

        # Compute the emotion score for each row in the dataframe
        emotion_scores_binary[emotion] = df[presence_cols].all(axis=1)
        emotion_scores_binary[emotion] = emotion_scores_binary[emotion].fillna(0)

    # Create a new dataframe with the emotion scores
    emotion_df_binary = pd.DataFrame(emotion_scores_binary)
    emotion_df_binary = emotion_df_binary.replace({False: 0, True: 1})

    # Let's add timestamp and success on
    columns_of_interest = ['timestamp', 'success']
    df_temp = df[columns_of_interest]

    # Concatenate the columns from df2 with df1
    emotion_df_binary = pd.concat([df_temp, emotion_df_binary], axis=1)

    return emotion_df_binary


# Raw Variables for Emotional Expressivity!

openface_emoHardSoftPres_dict = apply_function_to_dict_list(openface_radius_dict, detect_emotions, method='OpenFace', emo_AUs=emo_AUs)

# key: (df_emohard, df_emosoft, df_emopres)


# This will help us get Raw Variables for Overall Expressivity!

# key: (df_emohard, df_emosoft, df_emopres)

openface_lowerHardSoftPres_dict = apply_function_to_dict_list(openface_radius_dict, detect_emotions, method='OpenFace', emo_AUs=emo_AUs, additional_filter=AU_lower)
openface_upperHardSoftPres_dict = apply_function_to_dict_list(openface_radius_dict, detect_emotions, method='OpenFace', emo_AUs=emo_AUs, additional_filter=AU_upper)



THRESHOLD = 0.4
hsemotion_radius_binarized = apply_function_to_dict_list(hsemotion_radius_dict, binarize_cols, threshold=THRESHOLD)
hsemotion_emo_stats = apply_function_to_dict_list(hsemotion_radius_binarized, analyze_emotion_events_v2, max_frame_gap=10, event_minimum_num_frames=12, method='HSE')
hsemotion_emo_stats_dict_list = fill_empty_dfs_lists(hsemotion_emo_stats)



hsemotion_emo_stats_dict = average_inner_dfs(hsemotion_emo_stats_dict_list)


opengraphau_emoPres_dict = apply_function_to_dict_list(opengraphau_radius_dict, detect_emotions_og, method='OpenGraphAU', emo_AUs=emo_AUs)


opengraphau_lowerPres_dict = apply_function_to_dict_list(opengraphau_radius_dict, detect_emotions_og, method='OpenGraphAU', emo_AUs=emo_AUs, additional_filter=AU_lower)
opengraphau_upperPres_dict = apply_function_to_dict_list(opengraphau_radius_dict, detect_emotions_og, method='OpenGraphAU', emo_AUs=emo_AUs, additional_filter=AU_upper)



def openface_combine_and_binarize(soft_hard):
    """
    Combine the middle and last dataframes from detect_emotions output,
    with columns for AU raw and binary values renamed appropriately.

    Parameters:
    - soft_hard: a list of two DataFrames:
      - emo_soft: DataFrame, the second output of detect_emotions, with AU raw values
      - emo_binary: DataFrame, the third output of detect_emotions, with AU binary values

    Returns:
    - combined_df: DataFrame, combined dataframe with emotion raw and binary values.
    """
    emo_soft, emo_binary = soft_hard

    # Drop 'timestamp' and 'success' columns from emo_binary to prevent duplication
    emo_binary = emo_binary.drop(['timestamp', 'success'], axis=1, errors='ignore')

    # Rename columns in emo_soft and emo_binary for clarity
    emo_soft_columns = {col: f"{col}_Raw" for col in emo_soft.columns if col not in ['success', 'timestamp', 'frame']}
    emo_binary_columns = {col: f"{col}_Binary" for col in emo_binary.columns if col not in ['success', 'timestamp', 'frame']}

    emo_soft_renamed = emo_soft.rename(columns=emo_soft_columns)
    emo_binary_renamed = emo_binary.rename(columns=emo_binary_columns)

    # Combine the dataframes
    combined_df = pd.concat([emo_soft_renamed, emo_binary_renamed], axis=1)

    return combined_df


def take_second_from_tuple(input):
    return input[1]


def take_second_third_from_tuple(input):
    return [input[1], input[2]]


# Dictionary of dictionary of just soft values
openface_emoSoft_dict = apply_function_to_dict_list(openface_emoHardSoftPres_dict, take_second_from_tuple)


# Dictionary of list of two dictionaries: soft, presence (binary)
openface_emoSoftPres_dict = apply_function_to_dict_list(openface_emoHardSoftPres_dict, take_second_third_from_tuple)


# OPENFACE - affect/emotions (longer term)
openface_binarized = apply_function_to_dict_list(openface_emoSoftPres_dict, openface_combine_and_binarize)
openface_emo_stats = apply_function_to_dict_list(openface_binarized, analyze_emotion_events_v2, max_frame_gap=10, event_minimum_num_frames=12, method='OF')
openface_emo_stats_dict_list = fill_empty_dfs_lists(openface_emo_stats)


# OPENFACE - Averaging across time windows!

openface_emo_stats_dict = average_inner_dfs(openface_emo_stats_dict_list)


def rename_columns(df):
    """
    Renames the columns in a DataFrame according to specified pattern.

    Args:
        df (pandas DataFrame): The DataFrame to rename columns.

    Returns:
        pandas DataFrame: The DataFrame with renamed columns.
    """

    # Copy the DataFrame
    df_copy = df.copy()

    # Define the mapping for renaming columns
    column_mapping = {
        '_r': 'int',
        '_c': 'pres'
    }

    # Function to rename the columns
    def rename_column(column_name):
        au_number = column_name[2:4]
        if au_number.endswith('_'):
          au_number = '0' + au_number[0:1]
        suffix = column_name[-2:]
        if suffix in column_mapping:
            return f'fac_au{au_number}{column_mapping[suffix]}'
        else:
            return column_name

    # Rename the columns in the copied DataFrame
    df_copy = df_copy.rename(columns=rename_column)

    return df_copy

def calculate_AU_statistics(df):
    # Initialize an empty dictionary to store the computed statistics
    stats = {'AU': [], 'pres_pct': [], 'int_mean': [], 'int_std': []}

    # Iterate over the AU columns
    for col in df.columns:
        if col.startswith('fac_au') and ('pres' in col):
            # Calculate the percentage of frames where AU is present
            pres_pct = df[col].mean() * 100
            # Extract the AU number
            AU = col.split('au')[1][0:2]
            # Calculate the mean and standard deviation of intensity for the AU
            int_mean = df[f'fac_au{AU}int'].mean()
            int_std = df[f'fac_au{AU}int'].std()

            # Add the statistics to the dictionary
            stats['AU'].append(AU)
            stats['pres_pct'].append(pres_pct)
            stats['int_mean'].append(int_mean)
            stats['int_std'].append(int_std)

    # Create a DataFrame from the dictionary of statistics
    stats_df = pd.DataFrame(stats)

    return stats_df

def calculate_AU_statistics_og(df):
    # Stats for ONLY binary columns!
    # Initialize an empty dictionary to store the computed statistics
    stats = {'AU': [], 'pres_pct': []}

    # Iterate over the AU columns
    for col in df.columns:
        if col.startswith('fac_au') and ('pres' in col):
            # Calculate the percentage of frames where AU is present
            pres_pct = df[col].mean() * 100
            # Extract the AU number
            AU = col.split('au')[1][0:2]

            # Add the statistics to the dictionary
            stats['AU'].append(AU)
            stats['pres_pct'].append(pres_pct)

    # Create a DataFrame from the dictionary of statistics
    stats_df = pd.DataFrame(stats)

    return stats_df

def force_convert_to_float(dictionary):
    """
    Forcefully convert all DataFrames in a nested dictionary structure to have their columns as floats.

    Args:
        dictionary (dict): The dictionary containing nested dictionaries with lists of DataFrames.

    Returns:
        dict: A modified copy of the dictionary with all DataFrames converted to float.
    """
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                new_df_list = [df.astype(float) for df in df_list]
                new_dict[split_time][outer_key][timestamp] = new_df_list
    return new_dict


# Raw Variables!
openface_radius_renamed_dict = apply_function_to_dict_list(openface_radius_dict, rename_columns)



# Derived Variables!
openface_au_derived_dict_list = apply_function_to_dict_list(openface_radius_renamed_dict, calculate_AU_statistics)
openface_au_derived_dict_list = fill_empty_dfs_lists(openface_au_derived_dict_list)
openface_au_derived_dict_list = force_convert_to_float(openface_au_derived_dict_list)

# OPENFACE - Averaging across time windows!

openface_au_derived_dict = average_inner_dfs(openface_au_derived_dict_list)



opengraphau_radius_renamed_dict = apply_function_to_dict_list(opengraphau_radius_dict, rename_columns)
opengraphau_au_derived_dict_list = apply_function_to_dict_list(opengraphau_radius_renamed_dict, calculate_AU_statistics_og)
opengraphau_au_derived_dict_list = force_convert_to_float(opengraphau_au_derived_dict_list)
# OPENGRAPHAU - Averaging across time windows!

opengraphau_au_derived_dict = average_inner_dfs(opengraphau_au_derived_dict_list)




def calculate_emotion_express_statistics(tuple_to_unpack):
    """
    Calculates statistics for each emotion in the given DataFrames.

    Args:
        tuple_to_unpack: 3-membered tuple that has:
          df_emo_inthard (pandas DataFrame): DataFrame with emotion intensity (hard) values.
          df_emo_intsoft (pandas DataFrame): DataFrame with emotion intensity (soft) values.
          df_emo_pres (pandas DataFrame): DataFrame with emotion presence values.

    Returns:
        pandas DataFrame: A DataFrame with statistics for each emotion.
    """
    df_emo_inthard, df_emo_intsoft, df_emo_pres = tuple_to_unpack
    stats = {'emotion': [], 'pres_pct': [], 'intsoft_mean': [], 'intsoft_std': [], 'inthard_mean': []}

    emotions = [col for col in df_emo_inthard.columns if col not in ['timestamp', 'success']]

    for emotion in emotions:
        pres_pct = (df_emo_pres[emotion] == 1).mean() * 100
        intsoft_mean = df_emo_intsoft[emotion].mean()
        intsoft_std = df_emo_intsoft[emotion].std()
        inthard_mean = df_emo_inthard[emotion].mean()

        stats['emotion'].append(emotion)
        stats['pres_pct'].append(pres_pct)
        stats['intsoft_mean'].append(intsoft_mean)
        stats['intsoft_std'].append(intsoft_std)
        stats['inthard_mean'].append(inthard_mean)

    stats_df = pd.DataFrame(stats)
    return stats_df

def calculate_ee_stats_og(df_emo_pres):
    """
    Calculates statistics for each emotion in the given DataFrame.

    Args:
        df_emo_pres (pandas DataFrame): DataFrame with emotion presence values.

    Returns:
        pandas DataFrame: A DataFrame with statistics for each emotion.
    """
    stats = {'emotion': [], 'pres_pct': []}

    emotions = [col for col in df_emo_pres.columns if col not in ['timestamp', 'success']]

    for emotion in emotions:
        pres_pct = (df_emo_pres[emotion] == 1).mean() * 100


        stats['emotion'].append(emotion)
        stats['pres_pct'].append(pres_pct)

    stats_df = pd.DataFrame(stats)
    return stats_df

def calculate_ee_stats_hse(df, threshold):
    """
    Calculates statistics for each emotion in the given DataFrame.

    Args:
    df with emotion intensities for every video frame
    threshold for presence of emotion (i.e. 0.5)

    Returns:
        pandas DataFrame: A DataFrame with statistics for each emotion.
    """
    df_emo_intsoft = df
    stats = {'emotion': [], 'pres_pct': [], 'intsoft_mean': [], 'intsoft_std': []}

    emotions = [col for col in df_emo_intsoft.columns if col not in ['timestamp', 'success']]

    for emotion in emotions:
        pres_pct = (df_emo_intsoft[emotion] >= threshold).mean() * 100
        intsoft_mean = df_emo_intsoft[emotion].mean()
        intsoft_std = df_emo_intsoft[emotion].std()

        stats['emotion'].append(emotion)
        stats['pres_pct'].append(pres_pct)
        stats['intsoft_mean'].append(intsoft_mean)
        stats['intsoft_std'].append(intsoft_std)

    stats_df = pd.DataFrame(stats)
    return stats_df




# Derived Variables for Emotional Expressivity
openface_ee_derived_dict_list = apply_function_to_dict_list(openface_emoHardSoftPres_dict, calculate_emotion_express_statistics)
openface_ee_derived_dict = average_inner_dfs(openface_ee_derived_dict_list)


opengraphau_ee_derived_dict_list = apply_function_to_dict_list(opengraphau_emoPres_dict, calculate_ee_stats_og)
opengraphau_ee_derived_dict = average_inner_dfs(opengraphau_ee_derived_dict_list)


hsemotion_ee_derived_dict_list = apply_function_to_dict_list(hsemotion_radius_dict, calculate_ee_stats_hse, threshold=0.5)
hsemotion_ee_derived_dict = average_inner_dfs(hsemotion_ee_derived_dict_list)



def compute_oe_raw_vars(regular_tuple, lower_tuple, upper_tuple):
    # Takes in 3 3-membered tuples, each of which should be hardSoftPres
    # regular, lower, upper

    # Outputs one df with the raw variables for overall expressivity

    df_emo_inthard, df_emo_intsoft, df_emo_pres = regular_tuple
    df_emo_inthard_lower, df_emo_intsoft_lower, df_emo_pres_lower = lower_tuple
    df_emo_inthard_upper, df_emo_intsoft_upper, df_emo_pres_upper = upper_tuple

    df_emo_inthard = df_emo_inthard.drop(columns=['timestamp'])
    df_emo_intsoft = df_emo_intsoft.drop(columns=['timestamp'])
    df_emo_pres = df_emo_pres.drop(columns=['timestamp'])

    df_emo_inthard_lower = df_emo_inthard_lower.drop(columns=['timestamp'])
    df_emo_intsoft_lower = df_emo_intsoft_lower.drop(columns=['timestamp'])
    df_emo_pres_lower = df_emo_pres_lower.drop(columns=['timestamp'])

    df_emo_inthard_upper = df_emo_inthard_upper.drop(columns=['timestamp'])
    df_emo_intsoft_upper = df_emo_intsoft_upper.drop(columns=['timestamp'])
    df_emo_pres_upper = df_emo_pres_upper.drop(columns=['timestamp'])

    # Calculate the average values for emo_intsoft and emo_inthard across all frames
    avg_emo_intsoft = df_emo_intsoft.mean(axis=1)
    avg_emo_inthard = df_emo_inthard.mean(axis=1)

    # Calculate lower and upper averages across all frames
    avg_emo_intsoft_lower = df_emo_intsoft_lower.mean(axis=1)
    avg_emo_inthard_lower = df_emo_inthard_lower.mean(axis=1)
    avg_emo_intsoft_upper = df_emo_intsoft_upper.mean(axis=1)
    avg_emo_inthard_upper = df_emo_inthard_upper.mean(axis=1)

    # Create a new dataframe with the computed statistics
    stats_df = pd.DataFrame({'comintsoft': avg_emo_intsoft, 'cominthard': avg_emo_inthard,
                             'comlowintsoft': avg_emo_intsoft_lower, 'comlowinthard': avg_emo_inthard_lower,
                             'comuppintsoft': avg_emo_intsoft_upper, 'comuppinthard': avg_emo_inthard_upper,})

    return stats_df

def apply_function_to_dict_three(d1, d2, d3, func, **kwargs):
    """
    Apply a function that takes in 3 dfs and return a modified dictionary

    Args:
        d1, d2, d3: The dictionaries containing DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict_final: A modified copy of the dictionary with the function applied to each DataFrame.
    """
    dict_final = {}
    for key in d1.keys():
      dict_final[key] = func(d1[key], d2[key], d3[key], **kwargs)

    return dict_final

def apply_function_to_dict_three_list(d1, d2, d3, func, **kwargs):
    """
    Apply a function that takes in 3 dfs and return a modified dictionary

    Args:
        d1, d2, d3: The dictionaries containing LISTS of DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict_final: A modified copy of the dictionary with the function applied to each DataFrame in each list!
    """
    dict_final = {}
    for key in d1.keys():
      num_in_list = len(d1[key])
      list_building = []
      for i in range(num_in_list):
        list_building.append(func(d1[key][i], d2[key][i], d3[key][i], **kwargs))
      
      dict_final[key] = list_building
        

    return dict_final

def calculate_oe_summary_statistics(df):
    # Compute comintsoft_pct
    comintsoft_pct = (df['comintsoft'] > 0).mean() * 100

    # Compute comintsoft_mean and comintsoft_std
    comintsoft_mean = df['comintsoft'].mean()
    comintsoft_std = df['comintsoft'].std()

    # Compute cominthard_mean and cominthard_std
    cominthard_mean = df['cominthard'].mean()
    cominthard_std = df['cominthard'].std()

    # Compute comlowintsoft_pct
    comlowintsoft_pct = (df['comlowintsoft'] > 0).mean() * 100

    # Compute comlowintsoft_mean and comlowintsoft_std
    comlowintsoft_mean = df['comlowintsoft'].mean()
    comlowintsoft_std = df['comlowintsoft'].std()

    # Compute comuppinthard_mean and comuppinthard_std
    comuppinthard_mean = df['comuppinthard'].mean()
    comuppinthard_std = df['comuppinthard'].std()

    # Create a new DataFrame with the summary statistics
    summary_df = pd.DataFrame({
        'comintsoft_pct': [comintsoft_pct],
        'comintsoft_mean': [comintsoft_mean],
        'comintsoft_std': [comintsoft_std],
        'cominthard_mean': [cominthard_mean],
        'cominthard_std': [cominthard_std],
        'comlowintsoft_pct': [comlowintsoft_pct],
        'comlowintsoft_mean': [comlowintsoft_mean],
        'comlowintsoft_std': [comlowintsoft_std],
        'comuppinthard_mean': [comuppinthard_mean],
        'comuppinthard_std': [comuppinthard_std]
    })

    return summary_df


# Raw Variables for Overall Expressivity!

#openface_oe_raw = apply_function_to_dict_three(openface_emoHardSoftPres, openface_lowerHardSoftPres, openface_upperHardSoftPres, compute_oe_raw_vars)
#opengraphau_oe_raw = apply_function_to_dict_three(opengraphau_emoHardSoftPres, opengraphau_lowerHardSoftPres, opengraphau_upperHardSoftPres, compute_oe_raw_vars)


openface_oe_raw_dict_list = {}


# Loop through the dictionaries and sample one item from each with the same key
for time_now in openface_emoHardSoftPres_dict.keys():
    openface_oe_raw_dict_list[time_now] = {}
    for key in openface_emoHardSoftPres_dict[time_now].keys():
        openface_emo = openface_emoHardSoftPres_dict[time_now][key]
        openface_lower = openface_lowerHardSoftPres_dict[time_now][key]
        openface_upper = openface_upperHardSoftPres_dict[time_now][key]
    
        # Call the compute_oe_raw_vars function with the sampled items
        openface_oe_raw_dict_list[time_now][key] = apply_function_to_dict_three_list(openface_emo, openface_lower, openface_upper, compute_oe_raw_vars)


# Derived Variables for Overall Expressivity!

openface_oe_derived_dict_list = apply_function_to_dict_list(openface_oe_raw_dict_list, calculate_oe_summary_statistics)
openface_oe_derived_dict = average_inner_dfs(openface_oe_derived_dict_list)




def process_head_movement(df):
    # Ensure the pose columns are floats
    pose_cols = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    df[pose_cols] = df[pose_cols].astype(float)
    
    # Calculate Euclidean head movement (displacement)
    df['mov_headvel'] = np.sqrt(df[['pose_Tx', 'pose_Ty', 'pose_Tz']].diff().fillna(0).pow(2).sum(axis=1))
    
    # Assign frame-wise pitch, yaw, and roll directly from pose_Rx, pose_Ry, pose_Rz
    df['mov_hposepitch'] = df['pose_Rx']
    df['mov_hposeyaw'] = df['pose_Ry']
    df['mov_hposeroll'] = df['pose_Rz']
    
    # Calculate angular head movement using diff for pose_Rx, pose_Ry, pose_Rz, then take Euclidean norm
    df['mov_hposedist'] = np.sqrt(df[['pose_Rx', 'pose_Ry', 'pose_Rz']].diff().fillna(0).pow(2).sum(axis=1))
    
    # Calculate mean and std for the new variables
    output_dict = {}
    variables = ['mov_headvel', 'mov_hposepitch', 'mov_hposeyaw', 'mov_hposeroll', 'mov_hposedist']
    for var in variables:
        output_dict[f"{var}_mean"] = df[var].mean()
        output_dict[f"{var}_std"] = df[var].std()
    
    # Create output DataFrame from the output_dict
    output_df = pd.DataFrame([output_dict])
    
    return output_df

# Derived Variables for Head Movement!

openface_hm_derived_dict_list = apply_function_to_dict_list(openface_extras_radius_dict, process_head_movement)
# OPENFACE - Averaging across time windows!

openface_hm_derived_dict = average_inner_dfs(openface_hm_derived_dict_list)



def process_gaze_data(df):
    # Ensure all gaze-related columns are floats
    gaze_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
    df[gaze_cols] = df[gaze_cols].astype(float)
    
    # Initialize output dictionary
    output_dict = {}
    
    # Mapping for renaming
    rename_map = {
        'gaze_0_x': 'righteyex', 'gaze_0_y': 'righteyey', 'gaze_0_z': 'righteyez',
        'gaze_1_x': 'lefteyex', 'gaze_1_y': 'lefteyey', 'gaze_1_z': 'lefteyez'
    }
    
    # Calculate mean and std for each gaze direction component and rename
    for col in gaze_cols:
        new_base_name = rename_map[col]
        output_dict[f"mov_{new_base_name}_mean"] = df[col].mean()
        output_dict[f"mov_{new_base_name}_std"] = df[col].std()
    
    # Calculate Euclidean displacement for each eye in each frame
    df['mov_leyedisp'] = np.sqrt((df['gaze_1_x'].diff()**2 + df['gaze_1_y'].diff()**2 + df['gaze_1_z'].diff()**2).fillna(0))
    df['mov_reyedisp'] = np.sqrt((df['gaze_0_x'].diff()**2 + df['gaze_0_y'].diff()**2 + df['gaze_0_z'].diff()**2).fillna(0))
    
    # Add mean and std for the Euclidean displacements to output dict
    output_dict['mov_leyedisp_mean'] = df['mov_leyedisp'].mean()
    output_dict['mov_leyedisp_std'] = df['mov_leyedisp'].std()
    output_dict['mov_reyedisp_mean'] = df['mov_reyedisp'].mean()
    output_dict['mov_reyedisp_std'] = df['mov_reyedisp'].std()
    
    # Create output DataFrame from the output_dict
    output_df = pd.DataFrame([output_dict])
    
    return output_df


# Derived Variables for Head Movement!

openface_eg_derived_dict_list = apply_function_to_dict_list(openface_extras_radius_dict, process_gaze_data)

openface_eg_derived_dict = average_inner_dfs(openface_eg_derived_dict_list)



from scipy.spatial.distance import euclidean

def compute_ear(row):
    # Right eye
    try:
        d1 = euclidean((row['eye_lmk_X_10'], row['eye_lmk_Y_10'], row['eye_lmk_Z_10']), (row['eye_lmk_X_18'], row['eye_lmk_Y_18'], row['eye_lmk_Z_18']))
        d2 = euclidean((row['eye_lmk_X_12'], row['eye_lmk_Y_12'], row['eye_lmk_Z_12']), (row['eye_lmk_X_16'], row['eye_lmk_Y_16'], row['eye_lmk_Z_16']))
        d3 = euclidean((row['eye_lmk_X_8'], row['eye_lmk_Y_8'], row['eye_lmk_Z_8']), (row['eye_lmk_X_14'], row['eye_lmk_Y_14'], row['eye_lmk_Z_14']))
        right_ear = (d1 + d2) / (2.0 * d3)
    except:
        right_ear = 0 # Need some default value if the lmk values are infinity or 0
        
    # Left eye
    try:
        d4 = euclidean((row['eye_lmk_X_38'], row['eye_lmk_Y_38'], row['eye_lmk_Z_38']), (row['eye_lmk_X_46'], row['eye_lmk_Y_46'], row['eye_lmk_Z_46']))
        d5 = euclidean((row['eye_lmk_X_40'], row['eye_lmk_Y_40'], row['eye_lmk_Z_40']), (row['eye_lmk_X_44'], row['eye_lmk_Y_44'], row['eye_lmk_Z_44']))
        d6 = euclidean((row['eye_lmk_X_36'], row['eye_lmk_Y_36'], row['eye_lmk_Z_36']), (row['eye_lmk_X_42'], row['eye_lmk_Y_42'], row['eye_lmk_Z_42']))
        left_ear = (d4 + d5) / (2.0 * d6)
    except:
        left_ear = 0 # Need some default value if the lmk values are infinity or 0
    
    # Overall EAR
    return (right_ear + left_ear) / 2.0

def ebb_process_video_df(df):
    
    # Calculate EAR for each frame
    df['EAR'] = df.apply(compute_ear, axis=1)
    
    # Identify frames where a blink occurs
    df['is_blink'] = (df['EAR'] < 0.2) & (df['EAR'].shift(1) >= 0.2)
    
    # For each blink, find the timestamp difference to the previous blink
    blink_timestamps = df[df['is_blink']]['timestamp']
    mov_blinkdur = blink_timestamps.diff().fillna(0)  # This calculates the time between blinks
    
    # Convert to float
    try:
        mov_blinkdur = mov_blinkdur.apply(lambda x: x.total_seconds() if isinstance(x, pd.Timedelta) else x).astype(float)

        # Filter out blink durations over 10 seconds
        mov_blinkdur.loc[mov_blinkdur < 10]
    except:
        mov_blinkdur = 0

    
    
    
    # Initialize all values at zero
    features = {
        'mov_blink_ear_mean': 0,
        'mov_blink_ear_std': 0,
        'mov_blink_count': 0,
        'mov_blinkdur_mean': 0,
        'mov_blinkdur_std': 0
    }
    
    
    if df['is_blink'].sum() > 0:
        # Calculate requested features
        blink_ear_values = df[df['is_blink']]['EAR']
        features = {
            'mov_blink_ear_mean': blink_ear_values.mean(),
            'mov_blink_ear_std': 0 if np.isnan(blink_ear_values.std()) else blink_ear_values.std(),
            'mov_blink_count': df['is_blink'].sum(),
            'mov_blinkdur_mean': mov_blinkdur.mean(),
            'mov_blinkdur_std': 0 if np.isnan(mov_blinkdur.std()) else mov_blinkdur.std(),
        }
    
    return pd.DataFrame([features])

# Derived Variables for Eye Blink Behavior!

openface_ebb_derived_dict_list = apply_function_to_dict_list(openface_extras_radius_dict, ebb_process_video_df)

openface_ebb_derived_dict = average_inner_dfs(openface_ebb_derived_dict_list)



def fl_process_video_df(df):
    # Preparing the column names for X, Y, Z coordinates
    x_cols = [f'X_{i}' for i in range(68)]
    y_cols = [f'Y_{i}' for i in range(68)]
    z_cols = [f'Z_{i}' for i in range(68)]
    
    # Calculating the displacement for each landmark across frames
    disp_cols = []
    for x_col, y_col, z_col in zip(x_cols, y_cols, z_cols):
        disp_col = f'{x_col}_disp'
        df[disp_col] = np.sqrt((df[x_col].diff() ** 2) + (df[y_col].diff() ** 2) + (df[z_col].diff() ** 2))
        disp_cols.append(disp_col)
    
    # Calculating the mean and standard deviation of displacements for each landmark
    output_df = pd.DataFrame()
    for col in disp_cols:
        landmark_num = col.split('_')[1]
        output_df[f'fac_lmk{landmark_num}disp_mean'] = [df[col].mean()]
        output_df[f'fac_lmk{landmark_num}disp_std'] = [df[col].std()]
    
    # Return a DataFrame with calculated mean and standard deviation for each landmark displacement
    return output_df


# Derived Variables for Facial Landmark!

openface_fl_derived_dict_list = apply_function_to_dict_list(openface_extras_radius_dict, fl_process_video_df)


# OPENFACE - Averaging across time windows!

openface_fl_derived_dict = average_inner_dfs(openface_fl_derived_dict_list)




def calculate_fac_tremor(df, window_size=5):
    # Pad the DataFrame at the beginning and end to handle edge cases
    df_padded = pd.concat([df.iloc[:window_size-1].copy(), df, df.iloc[-window_size+1:].copy()]).reset_index(drop=True)
    
    # Initialize a DataFrame to hold the median tremor values for each landmark
    tremor_medians = pd.DataFrame()
    
    for i in range(68):  # For each landmark
        # Prepare column names
        x_col = f'X_{i}'
        y_col = f'Y_{i}'
        z_col = f'Z_{i}'
        
        # Calculate rolling mean positions
        rolling_means = df_padded[[x_col, y_col, z_col]].rolling(window=window_size, center=True).mean()
        
        # Calculate Euclidean distance from each frame's position to the rolling mean position
        distances = np.sqrt((df_padded[x_col] - rolling_means[x_col])**2 + 
                            (df_padded[y_col] - rolling_means[y_col])**2 + 
                            (df_padded[z_col] - rolling_means[z_col])**2)
        
        # Calculate median of distances for each window
        tremor_median = distances.rolling(window=window_size, center=True).median()
        
        # Append the median tremor value for the landmark to the tremor_medians DataFrame
        tremor_medians[f'fac_tremor_median_{i+1:02d}'] = tremor_median
        
    # Calculate the mean of median tremors across all frames for each landmark
    output_df = tremor_medians.mean().rename(lambda x: f'{x}_mean').to_frame().transpose()
    
    # Adjust the DataFrame to start from the original index
    output_df.index = [0]
    
    return output_df

# Derived Variables for Facial Tremor!

openface_ft_derived_dict_list = apply_function_to_dict_list(openface_extras_radius_dict, calculate_fac_tremor)


# OPENFACE - Averaging across time windows!

openface_ft_derived_dict = average_inner_dfs(openface_ft_derived_dict_list)



def calculate_pain_expressivity(df):
    # Calculate fac_paiintsoft for each frame
    soft_columns = ["AU04_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU20_r", "AU26_r"]
    df['fac_paiintsoft'] = df[soft_columns].mean(axis=1) / 5
    
    # Calculate fac_paiinthard for each frame
    hard_columns = ["AU04_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c", "AU20_c", "AU26_c"]
    df['fac_paiinthard'] = np.where(df[hard_columns].min(axis=1) > 0, df['fac_paiintsoft'], 0)
    
    # Calculate overall features
    results = {
        'fac_paiintsoft_pct': (df[hard_columns] > 0).any(axis=1).mean(),
        'fac_paiintsoft_mean': df['fac_paiintsoft'].mean(),
        'fac_paiintsoft_std': df['fac_paiintsoft'].std(),
        'fac_paiinthard_mean': df['fac_paiinthard'].mean(),
        'fac_paiinthard_std': df['fac_paiinthard'].std()
    }

    # Ensure no NaNs - replace NaNs with 0 for aggregation metrics
    results = {k: 0 if pd.isna(v) else v for k, v in results.items()}

    # Return results as a DataFrame
    return pd.DataFrame([results])

# Derived Variables for Pain Expressivity!

openface_pe_derived_dict_list = apply_function_to_dict_list(openface_radius_dict, calculate_pain_expressivity)


# OPENFACE - Averaging across time windows!

openface_pe_derived_dict = average_inner_dfs(openface_pe_derived_dict_list)





## Dictionary of list of relevant dictionaries
openface_dict_list_dict = {}

for time_split in openface_au_derived_dict.keys():
  openface_dict_list_dict[time_split] = {}
  for time_window in openface_au_derived_dict[time_split].keys():
      openface_dict_list_dict[time_split][time_window] = [ openface_au_derived_dict[time_split][time_window], openface_emo_stats_dict[time_split][time_window], 
                                                          openface_ee_derived_dict[time_split][time_window], openface_oe_derived_dict[time_split][time_window], 
                                                          openface_hm_derived_dict[time_split][time_window], openface_eg_derived_dict[time_split][time_window],
                                                          openface_ebb_derived_dict[time_split][time_window], openface_fl_derived_dict[time_split][time_window], 
                                                          openface_ft_derived_dict[time_split][time_window], openface_pe_derived_dict[time_split][time_window] ]



## Dictionary of list of relevant dictionaries
opengraphau_dict_list_dict = {}

for time_split in opengraphau_au_derived_dict.keys():
  opengraphau_dict_list_dict[time_split] = {}
  for time_window in opengraphau_au_derived_dict[time_split].keys():
      opengraphau_dict_list_dict[time_split][time_window] = [ opengraphau_au_derived_dict[time_split][time_window], opengraphau_ee_derived_dict[time_split][time_window] ]
      



## Dictionary of list of relevant dictionaries
hsemotion_dict_list_dict = {}

for time_split in hsemotion_emo_stats_dict.keys():
  hsemotion_dict_list_dict[time_split] = {}
  for time_window in hsemotion_emo_stats_dict[time_split].keys():
      hsemotion_dict_list_dict[time_split][time_window] = [ hsemotion_emo_stats_dict[time_split][time_window], hsemotion_ee_derived_dict[time_split][time_window] ]
      



opengraphau_dict_list_dict = opengraphau_dict_list_dict[5]
openface_dict_list_dict = openface_dict_list_dict[5]
hsemotion_dict_list_dict = hsemotion_dict_list_dict[5]

def partial_combine_dictionaries(dict1, dict2):
    # Takes element one (i.e. the AU matrix) from dict1, and all of dict2 (i.e. HSEmotion)
    combined_dict = {}

    for key in dict1:
        combined_dict[key] = [dict1[key][0]] + dict2[key]

    return combined_dict


ogauhsemotion_dict_list_dict = partial_combine_dictionaries(opengraphau_dict_list_dict, hsemotion_dict_list_dict)


# SAVE VARIABLES - EMOTION & AFFECT

save_var(openface_dict_list_dict, forced_name=f'openface_dict_list_dict_{PAT_SHORT_NAME}')

save_var(opengraphau_dict_list_dict, forced_name=f'opengraphau_dict_list_dict_{PAT_SHORT_NAME}')

save_var(hsemotion_dict_list_dict, forced_name=f'hsemotion_dict_list_dict_{PAT_SHORT_NAME}')

# SAVE VARIABLES - EMOTION & AFFECT

save_var(ogauhsemotion_dict_list_dict, forced_name=f'ogauhsemotion_dict_list_dict_{PAT_SHORT_NAME}')

print('[LOG] Saved Processed Feature Vectors')


# LOAD VARIABLES - EMOTION & AFFECT

openface_dict_list_dict = load_var(f'openface_dict_list_dict_{PAT_SHORT_NAME}')

opengraphau_dict_list_dict = load_var(f'opengraphau_dict_list_dict_{PAT_SHORT_NAME}')

hsemotion_dict_list_dict = load_var(f'hsemotion_dict_list_dict_{PAT_SHORT_NAME}')


# LOAD VARIABLES - EMOTION & AFFECT

ogauhsemotion_dict_list_dict = load_var(f'ogauhsemotion_dict_list_dict_{PAT_SHORT_NAME}')


def flatten_dataframes_dict(dataframes_list):
    # Initialize an empty dictionary to store the flattened data for each key
    flattened_data_dict = {}

    # Define the columns to ignore
    ignore_columns = ['success', 'timestamp', 'AU', 'emotion']

    for dataframes_dict in dataframes_list:
       for key, df in dataframes_dict.items():
          # Filter out the columns to be ignored
          filtered_df = df.drop(columns=[col for col in ignore_columns if col in df.columns])

          # Flatten the data by converting each DataFrame into a 1D array
          flattened_array = filtered_df.select_dtypes(include=[np.number, int, float, complex, \
                                                                pd.Int64Dtype(), pd.Float64Dtype(), pd.Int32Dtype(), \
                                                                pd.Float32Dtype()]).values.flatten()

          # Convert the flattened array to NumPy array and store it in the dictionary
          if key in flattened_data_dict:
              flattened_data_dict[key] = np.concatenate((flattened_data_dict[key], flattened_array))
          else:
              flattened_data_dict[key] = np.array(flattened_array)

    return flattened_data_dict


openface_vectors_dict = {}

for key, openface_dict_list_now in openface_dict_list_dict.items():
  openface_vectors_dict[key] = flatten_dataframes_dict(openface_dict_list_now)



opengraphau_vectors_dict = {}

for key, opengraphau_dict_list_now in opengraphau_dict_list_dict.items():
  opengraphau_vectors_dict[key] = flatten_dataframes_dict(opengraphau_dict_list_now)


hsemotion_vectors_dict = {}

for key, hsemotion_dict_list_now in hsemotion_dict_list_dict.items():
  hsemotion_vectors_dict[key] = flatten_dataframes_dict(hsemotion_dict_list_now)


ogauhsemotion_vectors_dict = {}

for key, ogauhsemotion_dict_list_now in ogauhsemotion_dict_list_dict.items():
  ogauhsemotion_vectors_dict[key] = flatten_dataframes_dict(ogauhsemotion_dict_list_now)


def ts_to_str(timestamp):
    return timestamp.strftime('%-m/%-d/%Y %H:%M:%S')

def str_to_ts(string_now):
  temp_var = pd.to_datetime(pd.to_datetime(string_now).strftime('%d-%b-%Y %H:%M:%S'))
  return pd.Timestamp(temp_var)


def ts_to_str_save(timestamp):
    # shorter version bc xlsxwriter sheet name char limit
    return timestamp.strftime('%-m_%-d %H_%M')


## Save our vectors to excel sheets!

def get_dict_name(dictionary):
    namespace = globals()
    for name, obj in namespace.items():
        if isinstance(obj, dict) and obj is dictionary:
            return name
    return None

def save_dicts_to_excel(dict_list, output_path):
  # Create an Excel writer object
  writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

  # Iterate over the keys in the dictionaries
  for key in dict_list[0].keys():
      # Write each dataframe to a separate sheet with the corresponding key as the sheet name
      for enum, dict_now in enumerate(dict_list):
        name_var = f'Matrix_{enum}'
        sheet_name_starter = f'{ts_to_str_save(key)}_{name_var}'
        dict_now[key].to_excel(writer, sheet_name=sheet_name_starter[:31])

  # Save the Excel file
  writer.close()
  return



os.makedirs(FEATURE_VIS_PATH, exist_ok=True)

for i in opengraphau_dict_list_dict.keys():
  save_dicts_to_excel(openface_dict_list_dict[i], FEATURE_VIS_PATH + f'openface_{PAT_SHORT_NAME}_{int(i)}_minutes.xlsx')
  save_dicts_to_excel(opengraphau_dict_list_dict[i], FEATURE_VIS_PATH + f'opengraphau_{PAT_SHORT_NAME}_{int(i)}_minutes.xlsx')
  save_dicts_to_excel(hsemotion_dict_list_dict[i], FEATURE_VIS_PATH + f'hsemotion_{PAT_SHORT_NAME}_{int(i)}_minutes.xlsx')
  save_dicts_to_excel(ogauhsemotion_dict_list_dict[i], FEATURE_VIS_PATH + f'ogauhse_{PAT_SHORT_NAME}_{int(i)}_minutes.xlsx')


print('[LOG] Feature Extraction Complete')


import random

def set_seed(x=5):
  np.random.seed(x)
  random.seed(x)


set_seed()





from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut, GridSearchCV


# def linRegOneMetric(vectors_dict, y, randShuffle=False, do_lasso=False, do_ridge=False, alpha=1.0):
#   # runs simple linear regression via one-left-out
#   # vectors_dict -- dictionary mapping time radius (in minutes) to features
#   # y -- a numpy array with labels (self-reported metrics)
#   # randShuffle -- do we shuffle the self-report labels?
#   # if do_lasso, does lasso regression
#   # if do_ridge, does ridge regression. Overrides do_lasso
#   # alpha - this is the weighting of either lasso or ridge

#   # returns a dictionary with several results:
#   # scores -- dictionary mapping each time radius to list of MSEs from each one-left-out
#   # preds -- dictionary mapping each time radius to a list of each one-left-out model's prediction
#   # y -- returns y again for convenience
#   # models -- dictionary mapping each time radius to a list of each one-left-out trained model (simple linear regression)

#   scores = {}
#   preds = {}
#   models = {}

#   if randShuffle:
#     y_using = np.random.permutation(y)
#   else:
#     y_using = y

#   for i in vectors_dict.keys():
#     model = LinearRegression()
#     if do_lasso:
#       model = Lasso(alpha=alpha)
#     if do_ridge:
#       model = Ridge(alpha=alpha)

#     # Compute MSEs via scikitlearn cross_val_score
#     scores_temp = cross_val_score(model, vectors_dict[i], y_using, cv=vectors_dict[i].shape[0], scoring='neg_mean_squared_error')
#     scores[i] = -1 * scores_temp

#     # Predictions via cross_val_predict
#     preds[i] = cross_val_predict(model, vectors_dict[i], y_using, cv=vectors_dict[i].shape[0])

#     # Now we need to iterate through and actually save the models themselves, since cross_val_score doesn't let us do that!
#     models_i_building = []
#     for test_index in range(vectors_dict[i].shape[0]):

#       X_train = np.delete(vectors_dict[i], test_index, axis=0)

#       y_train = np.delete(y_using, test_index, axis=0)

#       model = LinearRegression()
#       if do_lasso:
#         model = Lasso(alpha=alpha)
#       if do_ridge:
#         model = Ridge(alpha=alpha)
#       model.fit(X_train, y_train)
#       models_i_building.append(model)

#     models[i] = models_i_building

#   return scores, preds, y, models

def linRegOneMetric(vectors_dict, y, randShuffle=False, do_lasso=False, do_ridge=False, alpha=1.0, ALPHAS_FOR_SEARCH=None, num_permutations=0):
    """
    Runs regression (LASSO/Ridge/Linear) with optional nested alpha search and permutation testing.

    Args:
        vectors_dict (dict): Dictionary mapping time radius to feature arrays (numpy arrays).
        y (np.array): Labels (self-reported metrics).
        randShuffle (bool): Shuffle labels for random testing (default False).
        do_lasso (bool): Use LASSO regression (default False).
        do_ridge (bool): Use Ridge regression (default False).
        alpha (float): Regularization strength (default 1.0).
        ALPHAS_FOR_SEARCH (list or np.array): List of alphas to search in nested cross-validation (optional).
        num_permutations (int): Number of permutations for testing (default 0).

    Returns:
        dict: scores (MSE for each sample for each time radius),
        dict: preds (predicted values for each sample for each time radius),
        np.array: y (original or shuffled labels, based on randShuffle),
        dict: models (trained model objects for each time radius).
    """
    if randShuffle:
        y = np.random.permutation(y)

    # Default alpha search grid if not provided
    if ALPHAS_FOR_SEARCH is None:
        ALPHAS_FOR_SEARCH = np.arange(0.1, 5.0, 0.2)

    scores = {}
    preds = {}
    models = {}

    for time_radius, X in vectors_dict.items():
        # Determine the model
        if do_lasso:
            model = Lasso()
            param_grid = {'alpha': ALPHAS_FOR_SEARCH}
        elif do_ridge:
            model = Ridge()
            param_grid = {'alpha': ALPHAS_FOR_SEARCH}
        else:
            raise ValueError("Only LASSO or Ridge regression is supported for nested alpha search.")

        # Nested cross-validation for alpha search using LOOCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        # Best model after alpha search
        best_model = grid_search.best_estimator_
        best_alpha = grid_search.best_params_['alpha']

        # Predictions using LOOCV
        preds[time_radius] = cross_val_predict(best_model, X, y, cv=LeaveOneOut())
        
        # MSE scores using LOOCV
        scores[time_radius] = -1 * cross_val_score(best_model, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')

        # Save the trained model (one per time radius)
        models[time_radius] = best_model

        print(f"Time Radius: {time_radius}, Best Alpha: {best_alpha}")

    # Optional: Permutation testing
    if num_permutations > 0:
        print(f"Running {num_permutations} permutations for statistical testing...")
        permuted_r_list = []
        for _ in range(num_permutations):
            y_shuffled = np.random.permutation(y)
            permuted_preds = cross_val_predict(best_model, X, y_shuffled, cv=LeaveOneOut())
            permuted_r, _ = pearsonr(y_shuffled, permuted_preds)
            permuted_r_list.append(permuted_r)

        # Calculate actual Pearson's R
        actual_r, _ = pearsonr(y, preds[time_radius])
        
        # P-value computation
        p_value = np.mean([abs(r) >= abs(actual_r) for r in permuted_r_list])
        print(f"Permutation Test Pearson's R: {actual_r:.4f}, P-value: {p_value:.4f}")

    return scores, preds, y, models


def plot_predictions(y, y_pred, randShuffleR=None, ax=None, time_rad=None, metric=None):
    # Makes one scatterplot with Pearson's R and p value on it
    # give it the randShuffle Pearson's R
    # if you want to display that on the plot

    # Compute Pearson's R
    pearson_corr, p_val = pearsonr(y, y_pred)

    # Create the scatter plot on the specified axes
    if ax is None:
        ax_original = None
        fig, ax = plt.subplots()
        # adjust fonts!
        text_font = 16
    else:
        ax_original = ax
        text_font = 16


    ax.scatter(y, y_pred, label='Predicted vs. True', s=24)



    # Add the correlation coefficient and p-value on the plot
    ax.text(0.05, 0.90, f'Pearson\'s R: {pearson_corr:.2f}', transform=ax.transAxes, fontsize=text_font)
    ax.text(0.05, 0.80, f'P Value: {p_val:.2f}', transform=ax.transAxes, fontsize=text_font)
    if not(randShuffleR is None):
      ax.text(0.05, 0.70, f'Random Shuffle R: {randShuffleR:.2f}', transform=ax.transAxes, fontsize=text_font)

    # Set labels and title
    ax.set_xlabel('Self-Reported Scores', fontsize=17)
    ax.set_ylabel('Predicted Scores', fontsize=17)

    if metric is None:
      title_starter = 'Predicted vs. True'
    else:
      title_starter = metric

    if time_rad is None:
      ax.set_title(f'{title_starter} Scores', fontsize=17)
    else:
      num_hrs = int(time_rad) / 60
      if num_hrs > 1:
        ax.set_title(f'{title_starter}, Time Window = {num_hrs} Hours', fontsize=15)
      else:
        ax.set_title(f'{title_starter}, Time Window = {num_hrs} Hour', fontsize=15)


    # Add the line of best fit
    sns.regplot(x=y, y=y_pred, ax=ax, line_kws={'color': 'red', 'linestyle': '--'}, label='Line of Best Fit')

    # Add the shaded region for the 95% confidence interval
    #sns.regplot(x=y, y=y_pred, ax=ax, scatter=False, ci=95, color='gray', label='95% Confidence Interval')

    # Adjust the font size of the tick labels on the axes
    ax.tick_params(axis='both', labelsize=18)

    ax.set_adjustable('box')

    #set aspect ratio to 1
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    if ax_original is None:
        #plt.show()
        return pearson_corr, p_val, fig
    else:
        return pearson_corr, p_val



def plot_scatterplots(preds_dict, y, overall_title, savepath, randShuffleR=None):

    plt.rcParams['lines.markersize'] = 6
    subplot_title_font = 16
    full_title_font = 24

    num_plots = len(list(preds_dict.keys()))
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    r_list = []
    p_list = []

    # Calculate the desired figure size for larger plot
    figsize = (28, 12)

    # Create subplots with auto aspect ratio
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    #axes.set_adjustable('box')

    if num_rows == 1:
      axes = axes.reshape((1, num_cols))

    # Flatten the axes array if necessary
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([axes]).reshape(1, 1)


    # Loop through the dictionaries
    for i, (key, y_preds) in enumerate(preds_dict.items()):
        y_list = np.array(y).astype(float)
        y_pred = np.array(y_preds).astype(float)
        #y_pred = np.array([i[0] for i in y_pred])

        # Get the subplot coordinates
        row = i // num_cols
        col = i % num_cols

        # Plot predictions on the subplot
        if randShuffleR is None:
          pearson_corr, p_val = plot_predictions(y_list, y_pred, randShuffleR=randShuffleR, ax=axes[row, col])
        else:
          pearson_corr, p_val = plot_predictions(y_list, y_pred, randShuffleR=randShuffleR[i], ax=axes[row, col])
        r_list.append(pearson_corr)
        p_list.append(p_val)

        num_hrs = int(key) / 60
        if num_hrs > 1:
          axes[row, col].set_title(f'Time Window = {num_hrs} Hours', fontsize=subplot_title_font)
        else:
          axes[row, col].set_title(f'Time Window = {num_hrs} Hour', fontsize=subplot_title_font)
        #axes[row, col].set_aspect('equal')

        # Remove x-axis and y-axis labels from subplots
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('')

        #axes[row, col].set_adjustable('box')

    # Add overall title
    fig.suptitle(overall_title, fontsize=30, y=1)

    # Set shared x-axis and y-axis labels
    fig.text(0.5, 0.00, 'Self-Reported Scores', ha='center', fontsize=full_title_font)
    fig.text(-0.01, 0.5, 'Predicted Scores', va='center', rotation='vertical', fontsize=full_title_font)

    # Adjust spacing and layout
    fig.tight_layout()

    plt.savefig(savepath, bbox_inches='tight')

    #plt.show()

    return r_list, p_list, fig


def make_mse_boxplot(scores, metric, savepath, ax=None, method_now='OpenFace'):
    # scores -- dictionary that maps time radius (mins) to list of MSEs from one-left-out
    # metric - e.g. Mood or Anxiety

    # Combine the data into a single array
    data = [MSE_list for MSE_list in list(scores.values())]

    # Set the font sizes
    plt.rcParams.update({'font.size': 15})

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Create a box plot of the data
    labels_now = [f'{int(key) / 60}' for key in scores.keys()]

    ax.boxplot(data, labels=labels_now, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    # Determine the highest 75th percentile value among the four entries
    max_value = np.max([np.percentile(entry, 75) for entry in data])

    # Set the y-axis range conditionally
    if max_value > 100:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, max_value)

    # Set the labels and title
    ax.set_xlabel('Time Window (Hours)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'{metric} Prediction via {method_now}', y=1.1)

    plt.xticks(rotation=45)

    plt.savefig(savepath, bbox_inches='tight')

    # Show the plot if fig is None
    if fig is not None:
        return fig

def make_r_barplot(r_list, time_radius_list, metric, savepath, ax=None, method_now='OpenFace'):
    plt.rcParams.update({'font.size': 15})

    x_labels = [f'{int(i) / 60}' for i in time_radius_list]

    if ax is None:
        original_ax = None
        fig, ax = plt.subplots()
    else:
        original_ax = ax

    ax.bar(x_labels, r_list)

    # Set the y-axis range
    ax.set_ylim(-0.5, 1)

    # Set the labels and title
    ax.set_xlabel('Time Window (Hours)')
    ax.set_ylabel("Pearson's R")
    ax.set_title(f'{metric} Prediction via {method_now}', y=1.1)

    plt.xticks(rotation=45)

    plt.savefig(savepath, bbox_inches='tight')

    # Show the plot if ax is None
    if original_ax is None:
        #plt.show()
        return fig


def get_label_from_index(index, spreadsheet_path=FEATURE_LABEL_PATH+'openface_0.5_hours.xlsx'):
    if 'experimental' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1", "Matrix_2", "Matrix_3", "Matrix_4", "Matrix_5", "Matrix_6", "Matrix_7", "Matrix_8", "Matrix_9"]
      row_label_cols = ["AU", "emotion", "emotion", None, None, None, None, None, None, None]
    elif 'hsemotion' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1"]
      row_label_cols = ["emotion", "emotion"]
    elif 'opengraphau' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1"]
      row_label_cols = ["AU", "emotion"]
    elif 'openface' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1", "Matrix_2", "Matrix_3"]
      row_label_cols = ["AU", "emotion", "emotion", None]
    elif 'ofauhse' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1"]
      row_label_cols = ["AU", "emotion"]
    elif 'ogauhse' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1", "Matrix_2"]
      row_label_cols = ["AU", "emotion", "emotion"]
    elif 'all' in spreadsheet_path:
      matrices = ["Matrix_0", "Matrix_1", "Matrix_2", "Matrix_3", "Matrix_4", "Matrix_5"]
      row_label_cols = ["AU", "emotion", "emotion", None, "AU", "emotion"]
    else:
      print('BUG IN THE CODE! CHECK get_label_from_index')
      print('spreadsheet path is ', spreadsheet_path)


    xls = pd.ExcelFile(spreadsheet_path)

    for i, matrix in enumerate(matrices):
        # Find the sheet ending with the current matrix name
        sheet_name = next((s for s in xls.sheet_names if s.endswith(matrix)), None)
        if sheet_name is not None:
            # Load the sheet into a DataFrame, with the first row as column names
            df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name, header=0)

            # Get the column labels from the DataFrame
            col_labels = [col_now for col_now in df.columns.tolist() if not(col_now in ["AU", "emotion", "Unnamed: 0"])]

            if not row_label_cols[i] == 'AU':
                if 'emotion' in df.columns:
                    row_labels = df['emotion'].tolist()
                else:
                    row_labels = df['Unnamed: 0'].tolist()
            else:
                row_labels = df['AU'].tolist()

            # Get the numerical entries in the sheet excluding columns "AU" and "emotion" and "Unnamed: 0"
            numerical_entries = df.loc[:, ~df.columns.isin(["AU", "emotion", "Unnamed: 0"])].values.flatten()
            numerical_entries = numerical_entries[~pd.isnull(numerical_entries)]

            # Check if the index is within the range of numerical entries
            if index < len(numerical_entries):
                # Find the label corresponding to the index
                row_index, col_index = divmod(index, len(col_labels))
                if row_label_cols[i] == 'AU':
                    return f"AU{row_labels[row_index]} {col_labels[col_index]}"
                else:
                    if f'{row_labels[row_index]}' == '0':
                        return f"{col_labels[col_index]}"
                    else:
                        return f"{col_labels[col_index]} {row_labels[row_index]}"

            else:
                index -= len(numerical_entries)

    # Return None if the index is out of range or no suitable sheets found
    print('BUG IN THE CODE! INDEX TOO LARGE! CHECK get_label_from_index')
    print('spreadsheet path is ', spreadsheet_path)
    return None


def getTopFeaturesfromWeights(model_list, spreadsheet_path=FEATURE_LABEL_PATH+'openface_2.0_hours.xlsx'):
  # given a list of linear regression models,
  # returns their top 5 features (on average) from just weights!

  coef_array = [model_now.coef_ for model_now in model_list]
  coef_avg = np.mean(coef_array, axis=0)

  top_5_features = np.argsort(np.abs(coef_avg))[::-1][:5]

  top_5_english = [get_label_from_index(feat_ind, spreadsheet_path=spreadsheet_path) for feat_ind in top_5_features]

  return top_5_english


def featureAblate(vectors_array, y, do_lasso=False, do_ridge=False):
  # runs one-left-out linear regression,
  # deleting one feature at a time to determine most important features

  # vectors_array -- numpy array of feature vectors
  # y -- self-reported labels (e.g. for Mood, Anxiety, or something else)
  # if do_lasso, does lasso regression
  # if do_ridge, does ridge regression. Overrides do_lasso

  # returns scores, prs
  # scores -- (n_features, n_timestamps) numpy array of MSEs
  # prs -- (n_features,) numpy vector of pearson's R

  num_features = vectors_array.shape[1]
  num_timestamps = vectors_array.shape[0]

  scores = np.zeros((num_features, num_timestamps))
  prs = np.zeros((num_features,))

  # loop through each feature (for openface, 0 through 144) and delete just that
  for deleteNow in range(num_features):
    data = np.delete(vectors_array, deleteNow, axis=1)

    # make into dictionary to feed into our lin reg function
    data = {'placeholder': data}

    scores_temp, preds, y, _ = linRegOneMetric(data, y, do_lasso=do_lasso, do_ridge=do_ridge)
    scores_temp = scores_temp['placeholder']
    preds = preds['placeholder']

    # save MSEs
    scores[deleteNow, :] =  scores_temp

    # compute and save Pearson's R
    pearson_corr, _ = pearsonr(y, preds)
    prs[deleteNow] = pearson_corr

  return scores, prs

def featureAblate2D(vectors_array, y):
  # runs one-left-out linear regression,
  # deleting TWO features at a time to determine most important features

  # vectors_array -- numpy array of feature vectors
  # y -- self-reported labels (e.g. for Mood, Anxiety, or something else)

  # returns prs
  # prs -- (n_features, n_features) numpy vector of pearson's R
  # Note: ALWAYS index into prs with first index LOWER than second!

  num_features = vectors_array.shape[1]

  prs = np.zeros((num_features,num_features))

  # loop through each feature (for openface, 0 through 144) and delete just that
  for deleteNow in range(num_features):
    # delete a second one!
    for secondDelete in range(deleteNow+1, num_features):
      data = np.delete(vectors_array, [deleteNow, secondDelete], axis=1)

      # make into dictionary to feed into our lin reg function
      data = {'placeholder': data}

      _, preds, _, _ = linRegOneMetric(data, y)
      preds = preds['placeholder']

      # compute and save Pearson's R
      pearson_corr, _ = pearsonr(y, preds)
      prs[deleteNow, secondDelete] = pearson_corr

  return prs

def featureAblate3D(vectors_array, y):
  # runs one-left-out linear regression,
  # deleting THREE features at a time to determine most important features

  # vectors_array -- numpy array of feature vectors
  # y -- self-reported labels (e.g. for Mood, Anxiety, or something else)

  # returns prs
  # prs -- (n_features, n_features, n_features) numpy vector of pearson's R
  # Note: ALWAYS index into prs with earlier indices LOWER than subsequent ones.

  num_features = vectors_array.shape[1]

  prs = np.zeros((num_features, num_features, num_features))

  # loop through each feature (for openface, 0 through 144) and delete just that
  for deleteNow in range(num_features):
    # delete a second one!
    for secondDelete in range(deleteNow+1, num_features):
      # delete a third one!
      for thirdDelete in range(secondDelete+1, num_features):
        data = np.delete(vectors_array, [deleteNow, secondDelete, thirdDelete], axis=1)

        # make into dictionary to feed into our lin reg function
        data = {'placeholder': data}

        _, preds, _, _ = linRegOneMetric(data, y)
        preds = preds['placeholder']

        # compute and save Pearson's R
        pearson_corr, _ = pearsonr(y, preds)
        prs[deleteNow, secondDelete, thirdDelete] = pearson_corr

  return prs

def plotFeatAbMSEs(feat_ab_scores, original_mse_list, metric, time_radius, savepath, top_n=5, ax=None, spreadsheet_path=FEATURE_LABEL_PATH+'openface_2.0_hours.xlsx'):
  # takes feat_ab_scores, a numpy array (n_features, n_timestamps) of MSEs
  # outputs box and whisker plot of top_n features for the model

  # procedure: get the top_n features with lowest mse averaged across timestamps
  # make a box and whisker plot with each feature on x axis and MSEs on y axis
  # for x axis labels, convert the index of each feature to english label
  # by calling get_label_from_index(feat_ind)


  # Get the average MSE across timestamps for each feature
  avg_mses = np.mean(feat_ab_scores, axis=1)

  # avg MSEs minus original_avg_MSE (make it difference!)
  avg_mses = avg_mses - np.mean(original_mse_list)

  # Get the indices of the top_n features with the highest difference in MSEs from original
  top_indices = np.argsort(avg_mses)[-top_n:]
  top_indices = top_indices[::-1]

  # Get the English labels for the top_n features
  top_labels = [get_label_from_index(ind, spreadsheet_path=spreadsheet_path) for ind in top_indices]

  # Get the MSE values for the top_n features
  top_mses = feat_ab_scores[top_indices]

  # Adjust so it's top mses minus original
  original_list_repeated = np.repeat(np.array(original_mse_list).reshape(1, -1), top_n, axis=0)
  top_mses = top_mses - original_list_repeated

  # Create a box and whisker plot
  if ax is None:
      original_ax = None
      fig, ax = plt.subplots()
  else:
      original_ax = ax
  ax.boxplot(top_mses.T, labels=top_labels, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

  # Rotate x-axis labels by 45 degrees
  ax.set_xticklabels(top_labels, rotation=45)

  # Set the axis labels
  ax.set_xlabel('Features')
  ax.set_ylabel('Ablated - Original MSEs')

  # Set the title
  num_hrs = int(time_radius) / 60
  if num_hrs > 1:
      ax.set_title(f'Top {top_n} Features: {metric}, Time Window = {num_hrs} Hours')
  else:
      ax.set_title(f'Top {top_n} Features: {metric}, Time Window = {num_hrs} Hour')

  plt.savefig(savepath, bbox_inches='tight')
  return top_indices, fig



  

def plotFeatAbPRs(feat_ab_prs, original_r_val, metric, time_radius, savepath, top_n=5, ax=None, spreadsheet_path=FEATURE_LABEL_PATH+'openface_2.0_hours.xlsx'):
  # takes feat_ab_prs, a numpy array (n_features, ) of Pearson's R vals post-ablation
  # outputs bar plot of top_n features TO REMOVE for the model

  # procedure: get the top_n features with highest pearson's R
  # make a bar plot with each feature on x axis and pearson's R from feat_ab_prs on y axis
  # for x axis labels, convert the index of each feature to english label
  # by calling get_label_from_index(feat_ind)

  # if ax is given, plot on ax. If ax=None, make new fig, ax


  # Get the top_n features with highest Pearson's R values
  top_features_indices = np.argsort(feat_ab_prs)[-top_n:]
  top_features_indices = top_features_indices[::-1]

  # Get the labels for the top_n features
  top_features_labels = [get_label_from_index(index, spreadsheet_path=spreadsheet_path) for index in top_features_indices]

  # Get the corresponding Pearson's R values for the top_n features
  top_features_prs = feat_ab_prs[top_features_indices]

  # Plot the bar plot
  if ax is None:
      fig, ax = plt.subplots()
  ax.bar(top_features_labels, top_features_prs)

  # Rotate x-axis labels by 45 degrees
  ax.set_xticklabels(top_features_labels, rotation=45)

  # Set plot title and axis labels
  # Set the title
  num_hrs = int(time_radius) / 60
  if num_hrs > 1:
      ax.set_title(f'Top {top_n} Features to Remove: {metric}, Time Window = {num_hrs} Hours')
  else:
      ax.set_title(f'Top {top_n} Features to Remove: {metric}, Time Window = {num_hrs} Hour')

  ax.set_xlabel("Features")
  ax.set_ylabel(f"Pearson's R (Original={round(original_r_val, 2)})")

  # Save the plot
  plt.savefig(savepath, bbox_inches='tight')



def find_max_indices(array, top_n):
    # Flatten the 2D array into a 1D array
    flattened_array = array.flatten()

    # Find the indices of the top n maximum values in the flattened array
    max_indices = np.argsort(flattened_array)[-top_n:][::-1]

    # Convert the flattened indices to the corresponding row and column indices in the original array
    row_indices, col_indices = np.unravel_index(max_indices, array.shape)

    # Combine the row and column indices into pairs
    index_combinations = list(zip(row_indices, col_indices))

    return index_combinations

def plot_feat_scatterplots(vectors_array, y, feat_ind_list, metric, savepath, spreadsheet_path=FEATURE_LABEL_PATH+'openface_2.0_hours.xlsx'):
    # for each feature, plot feature on x axis and self-report score on y axis
    # vectors_array is the array of feature vectors for ONE time radius
    # y - self-reports
    # feat_ind_list - list of the indices of the top features
    # metric -- e.g. Mood or Anxiety
    # savepath - where to save the figure

    plt.rcParams['lines.markersize'] = 15

    num_plots = len(feat_ind_list)
    num_cols = min([len(feat_ind_list), 4])
    num_rows = (num_plots + num_cols - 1) // num_cols

    r_list = []
    p_list = []

    # Calculate the desired figure size for larger plot
    figsize = (28, 12)

    # Create subplots with auto aspect ratio
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    #axes.set_adjustable('box')

    if num_rows == 1:
      axes = axes.reshape((1, num_cols))

    # Flatten the axes array if necessary
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([axes]).reshape(1, 1)


    # Loop through the dictionaries
    for enum, i in enumerate(feat_ind_list):
        x_list = vectors_array[:, i].astype(float)
        y_list = np.array(y).astype(float)

        #y_pred = np.array([i[0] for i in y_pred])

        # Get the subplot coordinates
        row = enum // num_cols
        col = enum % num_cols

        # Plot predictions on the subplot
        pearson_corr, p_val = plot_predictions(x_list, y_list, randShuffleR=None, ax=axes[row, col])


        axes[row, col].set_title(f'{metric} vs. {get_label_from_index(i, spreadsheet_path=spreadsheet_path)}', fontsize=24)


        # Redo x-axis and y-axis labels for subplot
        axes[row, col].set_xlabel(get_label_from_index(i, spreadsheet_path=spreadsheet_path), fontsize=24)
        axes[row, col].set_ylabel('')

        #axes[row, col].set_adjustable('box')

    # Add overall title
    fig.suptitle(f'Top {num_plots} Features for {metric}', fontsize=30, y=1.05)

    # Set shared y-axis label
    #fig.text(0.5, 0, f'Self-Reported {metric} Scores', ha='center', fontsize=24)
    fig.text(-0.01, 0.5, f'Self-Reported {metric} Scores', va='center', rotation='vertical', fontsize=24)

    # Adjust spacing and layout
    fig.tight_layout()

    plt.savefig(savepath, bbox_inches='tight')

    #plt.show()

    return r_list, p_list, fig


def extractOneMetric(metric, vectors_now, df_moodTracking=df_moodTracking, remove_outliers=False):
  # extracts the vectors needed for linear regression
  # e.g. Mood only, for all time windows
  # metric -- a string that is a self-report metric (ex. 'Mood')
  # vectors_now -- our feature vectors (all)
  # df_moodTracking -- load in and pre-process self-report google sheet

  # returns vectors_return and y
  # vectors_return -- a dictionary mapping time radius (in minutes) to features
  # y -- a numpy array with labels (self-reported metrics)


  y = df_moodTracking[metric].values.astype(float)
  y = np.array([float(y_now) for y_now in y])

  # # just valid indices (remove nan self-reports!)
  # valid_indices = ~pd.isna(y)
  # y = y[valid_indices]

  # Initially, set valid_indices to include all indices
  valid_indices = np.arange(len(y))

  # Step 1: Remove NaN values
  nan_mask = ~pd.isna(y)
  y = y[nan_mask]
  valid_indices = valid_indices[nan_mask]


  if remove_outliers:
    # Step 2: Remove outliers
    mean_y = np.mean(y)
    std_y = np.std(y)
    outlier_mask = (y >= mean_y - 2 * std_y) & (y <= mean_y + 2 * std_y)
    y = y[outlier_mask]
    valid_indices = valid_indices[outlier_mask]



  vectors_return = {}

  def modify_keys(dictionary):
    # Using dictionary comprehension to create a new dictionary
    # with keys that have spaces removed
    return {ts_to_str(key): value for key, value in dictionary.items()}

  # loop through the inpatient videos at each time window before each timestamp we're considering (e.g. 10 mins)
  for i in vectors_now.keys():
    vectors_now_dict = vectors_now[i]
    
    vectors_now_dict_fixed = modify_keys(vectors_now_dict)
    
    # TODO: Fix bug where some vectors at the end have more values than others!
    # vectors_one_timestamp = np.array([vectors_now_dict_fixed[fn] for fn in df_moodTracking['Datetime']])

    shapes = []
    for j in range(len(df_moodTracking['Datetime'])):
        shapes.append(vectors_now_dict_fixed[df_moodTracking['Datetime'][j]].shape[0])
    vector_proper_shape = np.min(shapes)
    
    vectors_one_timestamp = np.array([vectors_now_dict_fixed[fn][:vector_proper_shape] for fn in df_moodTracking['Datetime']])
    
    # we want just the valid features (where self-report is not nan)
    vectors_one_timestamp = vectors_one_timestamp[valid_indices]
    
    vectors_return[i] = vectors_one_timestamp
    
  return vectors_return, y



def plot_pearsons_r_vs_alpha(pearson_r_list, ALPHAS_FOR_SEARCH, method, save_path):
    """
    Plots Pearson's R values against Alphas and saves the plot to the specified path.

    :param pearson_r_list: List of Pearson's R values.
    :param ALPHAS_FOR_SEARCH: List of Alpha values.
    :param method: The method used (string).
    :param save_path: File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(ALPHAS_FOR_SEARCH, pearson_r_list, marker='o')
    plt.title(f'LASSO: Alpha Search {method}')
    plt.xlabel('Alpha')
    plt.ylabel("Pearson's R")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()





# GENERATE ALL PLOTS! ONE CODE BLOCK


#all_metrics = [col for col in df_moodTracking.columns if col != 'Datetime']
#all_metrics = ['Mood', 'Anxiety', 'Hunger']
all_metrics = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']


FILE_ENDING = '.png'
# RESULTS_PREFIX_LIST = ['OF_', 'OGAU_', 'OFAUHSE_', 'OGAUHSE_', 'HSE_', 'ALL_',
#                        'OF_L_', 'OGAU_L_', 'OFAUHSE_L_', 'OGAUHSE_L_', 'HSE_L_', 'ALL_L_',
#                        'OF_R_', 'OGAU_R_', 'OFAUHSE_R_', 'OGAUHSE_R_', 'HSE_R_', 'ALL_R_']

# RESULTS_PREFIX_LIST = ['OF_L_', 'OGAUHSE_L_', 'OGAU_L_', 'OFAUHSE_L_', 'HSE_L_', 'ALL_L_']

RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']


# Do we remove ground truth labels that are over 2 standard deviations from the mean?
REMOVE_OUTLIERS = False




for RESULTS_PREFIX in RESULTS_PREFIX_LIST:
    print(f'[LOG] Generating Output Plots: {RESULTS_PREFIX}')
    do_lasso = False
    do_ridge = False

    if '_L_' in RESULTS_PREFIX:
        do_lasso = True

    if '_R_' in RESULTS_PREFIX:
        do_ridge = True

    if 'OF_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + f'experimental3_openface_0.5_hours.xlsx'
        vectors_now = openface_vectors_dict
        method_now = 'OpenFace'

    elif 'OGAU_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + 'opengraphau_0.5_hours.xlsx'
        vectors_now = opengraphau_vectors_dict
        method_now = 'OpenGraphAU'

    elif 'OFAUHSE_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + 'ofauhse_0.5_hours.xlsx'
        vectors_now = ofauhsemotion_vectors_dict
        method_now = 'OFAU+HSE'

    elif 'OGAUHSE_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + 'ogauhse_0.5_hours.xlsx'
        vectors_now = ogauhsemotion_vectors_dict
        method_now = 'OGAU+HSE'

    elif 'HSE_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + 'hsemotion_0.5_hours.xlsx'
        vectors_now = hsemotion_vectors_dict
        method_now = 'HSEmotion'

    elif 'ALL_' in RESULTS_PREFIX:
        spreadsheet_path = FEATURE_LABEL_PATH + 'all_0.5_hours.xlsx'
        vectors_now = all_vectors_dict
        method_now = 'ALL(OF+OG+HSE)'

    # Let's put each setting in its own folder!
    os.makedirs(RESULTS_PATH_BASE + RESULTS_PREFIX, exist_ok=True)
    results_prefix_unmodified = RESULTS_PREFIX
    RESULTS_PREFIX = RESULTS_PREFIX + '/' + RESULTS_PREFIX

    # Create a dictionary to store predictions and true values
    predictions_dict = {}


    # Loop through metrics (Anxiety, Depression, Mood, etc.)
    for metric in all_metrics:
        print('METRIC NOW: ', metric)
        if do_lasso:
            # TODO: Add the alpha search back in!
            alpha_now = 1.0
            #alpha_now = best_alphas_lasso[results_prefix_unmodified].get(metric, 1.0)  # Use the specific alpha for the metric
        elif do_ridge:
            alpha_now = best_alphas_ridge[results_prefix_unmodified].get(metric, 1.0)  # Use the specific alpha for the metric
        else:
            # Neither lasso nor ridge, so alpha is irrelevant
            alpha_now = 1.0

        vectors_return, y = extractOneMetric(metric, vectors_now=vectors_now, remove_outliers=REMOVE_OUTLIERS)
        # scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
        # scores_r, preds_r, _, models_r = linRegOneMetric(vectors_return, y, randShuffle=True, alpha=alpha_now)

        # Run LASSO with nested alpha search and permutation testing
        scores, preds, y, models = linRegOneMetric(
            vectors_return, 
            y, 
            do_lasso=do_lasso, 
            do_ridge=do_ridge, 
            ALPHAS_FOR_SEARCH=np.arange(0.1, 5.0, 0.2),  # Provide the alpha search grid
            num_permutations=0  # Set to 0 if no permutation testing is needed
        )

        # Run permutation testing by enabling randShuffle and specifying num_permutations
        scores_r, preds_r, _, models_r = linRegOneMetric(
            vectors_return, 
            y, 
            randShuffle=True,  # Enable random shuffling for permutation testing
            do_lasso=do_lasso, 
            do_ridge=do_ridge, 
            ALPHAS_FOR_SEARCH=np.arange(0.1, 5.0, 0.2),  # Use the same alpha search grid
            num_permutations=100  # Number of permutations
        )

        # make scatterplots
        randShuffleR, _, _ = plot_scatterplots(preds_r, y, f'{metric} Random Shuffle', RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterRand{FILE_ENDING}')
        r_list, p_list, scatterFig = plot_scatterplots(preds, y, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterplots{FILE_ENDING}', randShuffleR=randShuffleR)

        # Determine our best time radius for this metric based on Pearson's R
        best_time_radius = list(scores.keys())[np.argmax(r_list)]
        best_mse_list = scores[best_time_radius]
        best_avg_mse = np.mean(scores[best_time_radius])
        best_pearson_r = r_list[np.argmax(r_list)]

        # bar plot for pearson r
        rPlotFig = make_r_barplot(r_list, list(scores.keys()), metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_R{FILE_ENDING}', method_now=method_now)

        # make MSE plot
        MSEPlotFig = make_mse_boxplot(scores, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_MSE{FILE_ENDING}', method_now=method_now)

        # Feature ablation
        # feat_ab_scores, feat_ab_prs = featureAblate(vectors_return[best_time_radius], y, do_lasso=do_lasso, do_ridge=do_ridge)

        # top_indices, featAbMSEFig = plotFeatAbMSEs(feat_ab_scores, best_mse_list, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_MSEs{FILE_ENDING}', spreadsheet_path=spreadsheet_path)
        # plotFeatAbPRs(feat_ab_prs, best_pearson_r, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_R{FILE_ENDING}', spreadsheet_path=spreadsheet_path)

        # extract just ONE scatterplot (the best pearson's R) and save it individually
        plt.rcParams['lines.markersize'] = 9
        _, _, bestScatterFig = plot_predictions(y, preds[best_time_radius], randShuffleR=randShuffleR[np.argmax(r_list)], ax=None, time_rad=best_time_radius, metric=metric)
        bestScatterFig.savefig(RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_bestScatter{FILE_ENDING}', bbox_inches='tight')

        # Plot top n features vs. self-reported scores
        # PLOT_NOW = 3
        # plot_feat_scatterplots(vectors_array=vectors_return[best_time_radius], y=y, feat_ind_list=top_indices[:PLOT_NOW], metric=metric, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_topFeats{FILE_ENDING}', spreadsheet_path=spreadsheet_path)

        # Store predictions and true values for each time radius
        predictions_dict[metric] = {
            'y_true': y,
            'preds': preds,
            'best_time_radius': best_time_radius,
            'randShuffleR': randShuffleR[np.argmax(r_list)],
        }

    # Save the predictions and true values for re-plotting later
    save_var(predictions_dict, forced_name=f'predictions_{PAT_SHORT_NAME}_{results_prefix_unmodified}')


print(f'[LOG] Feature Extraction & Plotting Complete: {PAT_SHORT_NAME}')

