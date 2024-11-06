# Assumes you have already run main_FeatureExtractionInpatient.py and you just need linear regression! 


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






# LOAD VARIABLES - EMOTION & AFFECT

openface_dict_list_dict = load_var(f'openface_dict_list_dict_{PAT_SHORT_NAME}')

opengraphau_dict_list_dict = load_var(f'opengraphau_dict_list_dict_{PAT_SHORT_NAME}')

hsemotion_dict_list_dict = load_var(f'hsemotion_dict_list_dict_{PAT_SHORT_NAME}')


# LOAD VARIABLES - EMOTION & AFFECT

ogauhsemotion_dict_list_dict = load_var(f'ogauhsemotion_dict_list_dict_{PAT_SHORT_NAME}')

print('[LOG] Loaded Processed Feature Vectors')


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


def linRegOneMetric(vectors_dict, y, randShuffle=False, do_lasso=False, do_ridge=False, alpha=1.0):
  # runs simple linear regression via one-left-out
  # vectors_dict -- dictionary mapping time radius (in minutes) to features
  # y -- a numpy array with labels (self-reported metrics)
  # randShuffle -- do we shuffle the self-report labels?
  # if do_lasso, does lasso regression
  # if do_ridge, does ridge regression. Overrides do_lasso
  # alpha - this is the weighting of either lasso or ridge

  # returns a dictionary with several results:
  # scores -- dictionary mapping each time radius to list of MSEs from each one-left-out
  # preds -- dictionary mapping each time radius to a list of each one-left-out model's prediction
  # y -- returns y again for convenience
  # models -- dictionary mapping each time radius to a list of each one-left-out trained model (simple linear regression)

  scores = {}
  preds = {}
  models = {}

  if randShuffle:
    y_using = np.random.permutation(y)
  else:
    y_using = y

  for i in vectors_dict.keys():
    model = LinearRegression()
    if do_lasso:
      model = Lasso(alpha=alpha)
    if do_ridge:
      model = Ridge(alpha=alpha)

    # Compute MSEs via scikitlearn cross_val_score
    scores_temp = cross_val_score(model, vectors_dict[i], y_using, cv=vectors_dict[i].shape[0], scoring='neg_mean_squared_error')
    scores[i] = -1 * scores_temp

    # Predictions via cross_val_predict
    preds[i] = cross_val_predict(model, vectors_dict[i], y_using, cv=vectors_dict[i].shape[0])

    # Now we need to iterate through and actually save the models themselves, since cross_val_score doesn't let us do that!
    models_i_building = []
    for test_index in range(vectors_dict[i].shape[0]):

      X_train = np.delete(vectors_dict[i], test_index, axis=0)

      y_train = np.delete(y_using, test_index, axis=0)

      model = LinearRegression()
      if do_lasso:
        model = Lasso(alpha=alpha)
      if do_ridge:
        model = Ridge(alpha=alpha)
      model.fit(X_train, y_train)
      models_i_building.append(model)

    models[i] = models_i_building

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

    index_orig = index
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
    print(f'Original index was {index_orig}')
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



# TODO: Add the alpha search back in!
# # ALPHA PARAMETER SEARCH FOR LASSO - RUN THIS FIRST!

# all_metrics = [col for col in df_moodTracking.columns if col != 'Datetime']

# FILE_ENDING = '.png'

# # We are just searching using lasso regression
# #RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OFAUHSE_L_', 'OGAUHSE_L_', 'HSE_L_', 'ALL_L_']
# RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
# #RESULTS_PREFIX_LIST = ['OGAUHSE_L_']


# EMOTIONS_FOR_SEARCH = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain'] # We are just searching on Mood
# TIME_WINDOW_FOR_SEARCH = '180' # We are just searching 3 hours

# # List of alpha values to search through
# #ALPHAS_FOR_SEARCH = np.arange(0, 1.6, 0.1)
# ALPHAS_FOR_SEARCH = np.arange(0, 5, 0.2)
# #ALPHAS_FOR_SEARCH = np.arange(0, 10, 0.2)

# # This will populate with the best alphas for each prefix in RESULTS_PREFIX_LIST
# best_alphas_lasso = {}

# for RESULTS_PREFIX in RESULTS_PREFIX_LIST:
#     do_lasso = False
#     do_ridge = False

#     if '_L_' in RESULTS_PREFIX:
#         do_lasso = True

#     if '_R_' in RESULTS_PREFIX:
#         do_ridge = True

#     if 'OF_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + f'experimental3_openface_0.5_hours.xlsx'
#         vectors_now = openface_vectors_dict
#         method_now = 'OpenFace'

#     elif 'OGAU_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + 'opengraphau_0.5_hours.xlsx'
#         vectors_now = opengraphau_vectors_dict
#         method_now = 'OpenGraphAU'

#     elif 'OFAUHSE_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + 'ofauhse_0.5_hours.xlsx'
#         vectors_now = ofauhsemotion_vectors_dict
#         method_now = 'OFAU+HSE'

#     elif 'OGAUHSE_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + 'ogauhse_0.5_hours.xlsx'
#         vectors_now = ogauhsemotion_vectors_dict
#         method_now = 'OGAU+HSE'

#     elif 'HSE_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + 'hsemotion_0.5_hours.xlsx'
#         vectors_now = hsemotion_vectors_dict
#         method_now = 'HSEmotion'

#     elif 'ALL_' in RESULTS_PREFIX:
#         spreadsheet_path = FEATURE_LABEL_PATH + 'all_0.5_hours.xlsx'
#         vectors_now = all_vectors_dict
#         method_now = 'ALL(OF+OG+HSE)'

#     # Let's put each setting in its own folder!
#     os.makedirs(RESULTS_PATH_BASE + 'SEARCH_Alpha_Lasso/' + RESULTS_PREFIX, exist_ok=True)
#     results_prefix_unmodified = RESULTS_PREFIX
#     RESULTS_PREFIX = 'SEARCH_Alpha_Lasso/' + RESULTS_PREFIX + '/' + RESULTS_PREFIX

#     # Initialize a dictionary to store the best alpha values for each emotion
#     best_alphas_lasso[results_prefix_unmodified] = {}

#     for metric in EMOTIONS_FOR_SEARCH:
#         print('METRIC NOW: ', metric)
#         pearson_r_list = []  # Reset the R list for each metric

#         for alpha_now in ALPHAS_FOR_SEARCH:

#             avg_best_R = 0

#             vectors_return, y = extractOneMetric(metric, vectors_now=vectors_now)

#             # Limit to just one time window for alpha search
#             tmp_vectors = vectors_return
#             vectors_return = {}
#             vectors_return[TIME_WINDOW_FOR_SEARCH] = tmp_vectors[TIME_WINDOW_FOR_SEARCH]
#             del tmp_vectors

#             scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
#             scores_r, preds_r, _, models_r = linRegOneMetric(vectors_return, y, randShuffle=True, alpha=alpha_now)

#             # make scatterplots
#             randShuffleR, _, _ = plot_scatterplots(preds_r, y, f'{metric} Random Shuffle', RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterRand_{alpha_now}{FILE_ENDING}')
#             r_list, p_list, scatterFig = plot_scatterplots(preds, y, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterplots_{alpha_now}{FILE_ENDING}', randShuffleR=randShuffleR)

#             # Determine our best time radius for this metric based on Pearson's R
#             best_time_radius = list(scores.keys())[np.argmax(r_list)]
#             best_mse_list = scores[best_time_radius]
#             best_avg_mse = np.mean(scores[best_time_radius])
#             best_pearson_r = r_list[np.argmax(r_list)]

#             # Add to our avg best R
#             avg_best_R = avg_best_R + best_pearson_r

#             # bar plot for pearson r
#             rPlotFig = make_r_barplot(r_list, list(scores.keys()), metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_R_{alpha_now}{FILE_ENDING}', method_now=method_now)

#             # make MSE plot
#             MSEPlotFig = make_mse_boxplot(scores, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_MSE_{alpha_now}{FILE_ENDING}', method_now=method_now)

#         # Add one R value for this alpha value to pearson_r_list
#         avg_best_R = avg_best_R / len(EMOTIONS_FOR_SEARCH)
#         pearson_r_list.append(avg_best_R)

#         # Plot R vs. alpha for this setting
#         plot_pearsons_r_vs_alpha(pearson_r_list=pearson_r_list, ALPHAS_FOR_SEARCH=ALPHAS_FOR_SEARCH, method=method_now, save_path=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_Alpha_Search{FILE_ENDING}')

#         # Find best alpha for this metric and store it
#         best_index_of_alpha = np.argmax(pearson_r_list)
#         best_alpha_value = ALPHAS_FOR_SEARCH[best_index_of_alpha]
#         best_alphas_lasso[results_prefix_unmodified][metric] = best_alpha_value


# # SAVE VARIABLES

# save_var(best_alphas_lasso, forced_name=f'best_alphas_lasso_{PAT_SHORT_NAME}')





# GENERATE ALL PLOTS! ONE CODE BLOCK

# TODO: Add the alpha search back in!
# if 'best_alphas_lasso' not in globals():
#     raise NameError("GO RUN THE LASSO ALPHA PARAMETER SEARCH BLOCK FIRST!")

# if 'best_alphas_ridge' not in globals():
#     raise NameError("GO RUN THE RIDGE ALPHA PARAMETER SEARCH BLOCK FIRST!")


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
        scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
        scores_r, preds_r, _, models_r = linRegOneMetric(vectors_return, y, randShuffle=True, alpha=alpha_now)

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
        feat_ab_scores, feat_ab_prs = featureAblate(vectors_return[best_time_radius], y, do_lasso=do_lasso, do_ridge=do_ridge)

        top_indices, featAbMSEFig = plotFeatAbMSEs(feat_ab_scores, best_mse_list, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_MSEs{FILE_ENDING}', spreadsheet_path=spreadsheet_path)
        plotFeatAbPRs(feat_ab_prs, best_pearson_r, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_R{FILE_ENDING}', spreadsheet_path=spreadsheet_path)

        # extract just ONE scatterplot (the best pearson's R) and save it individually
        plt.rcParams['lines.markersize'] = 9
        _, _, bestScatterFig = plot_predictions(y, preds[best_time_radius], randShuffleR=randShuffleR[np.argmax(r_list)], ax=None, time_rad=best_time_radius, metric=metric)
        bestScatterFig.savefig(RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_bestScatter{FILE_ENDING}', bbox_inches='tight')

        # Plot top n features vs. self-reported scores
        PLOT_NOW = 3
        plot_feat_scatterplots(vectors_array=vectors_return[best_time_radius], y=y, feat_ind_list=top_indices[:PLOT_NOW], metric=metric, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_topFeats{FILE_ENDING}', spreadsheet_path=spreadsheet_path)

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

