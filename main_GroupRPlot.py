import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Define the paths and prefixes
RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
PREFIX_1 = "OGAU_L_"
PREFIX_1_PAIN = "OGAUHSE_L_"  # Just for pain, we use AU + HSE
PREFIX_2 = "OF_L_"

LABEL_1 = "FaceDx"
LABEL_2 = "OpenFace"

METRICS = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']  # Add all metrics you're interested in

SHOW_PREFIX_2 = True  # Boolean flag to control whether PREFIX_2 should be included in the plots




# Ignore all warnings
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')

# Ensure the results directory exists
os.makedirs(RESULTS_PATH_BASE, exist_ok=True)

# Function to load variables
def load_var(variable_name, RUNTIME_VAR_PATH):
    with open(os.path.join(RUNTIME_VAR_PATH, f'{variable_name}.pkl'), 'rb') as file:
        return pickle.load(file)

# Function to detect all unique patient short names based on saved files
def detect_patients():
    files = os.listdir(RUNTIME_VAR_PATH)
    patient_names = set()

    for file in files:
        if file.startswith('predictions_S_'):
            # Extract the patient name assuming it starts with 'S_' and is followed by '_'
            parts = file.split('_')
            patient_name = '_'.join(parts[1:3])  # This ensures you capture the full patient name after 'predictions_'
            patient_names.add(patient_name)
    
    return list(patient_names)

# Function to check if a patient meets the inclusion criteria
def meets_inclusion_criteria(df, metric):
    if metric not in df.columns:
        return False
    values = df[metric].dropna().unique()
    return len(df[metric].dropna()) >= 5 and len(values) >= 4

# Load and preprocess the mood tracking sheet
def preprocess_mood_tracking(PAT_SHORT_NAME):
    MOOD_TRACKING_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
    df = pd.read_excel(MOOD_TRACKING_SHEET_PATH, sheet_name=f'{PAT_SHORT_NAME}')

    # Properly deal with the missing values
    df = df.replace('', np.nan).replace(' ', np.nan).fillna(value=np.nan)

    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%-m/%-d/%Y %H:%M:%S')

    # Drop unnecessary columns
    df = df.drop(columns=['Notes'], errors='ignore')

    return df

# Detect all patients
patients = detect_patients()

# Loop through each metric
for metric in METRICS:
    r_values_prefix_1 = []
    r_values_prefix_2 = []

    # Print header for the metric
    print(f"\nResults for {metric.capitalize()}:")

    # Loop through each patient
    for patient in patients:
        try:
            # Preprocess mood tracking for the patient
            df_moodTracking = preprocess_mood_tracking(patient)

            # Check if patient meets the inclusion criteria
            if not meets_inclusion_criteria(df_moodTracking, metric):
                print(f"{patient} excluded for {metric} due to not meeting inclusion criteria.")
                continue

            # Load the predictions for both prefixes
            if metric == 'Pain':
                predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1_PAIN}', RUNTIME_VAR_PATH)[f'{metric}']
            else:
                predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1}', RUNTIME_VAR_PATH)[f'{metric}']
            predictions_prefix_2 = load_var(f'predictions_{patient}_{PREFIX_2}', RUNTIME_VAR_PATH)[f'{metric}'] if SHOW_PREFIX_2 else None

            # Calculate Pearson's R for each
            r_1 = np.corrcoef(predictions_prefix_1['y_true'], predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']])[0, 1]
            r_2 = np.corrcoef(predictions_prefix_2['y_true'], predictions_prefix_2['preds'][predictions_prefix_2['best_time_radius']])[0, 1] if SHOW_PREFIX_2 else None

            # Exclude patient if either score is NaN
            if np.isnan(r_1) or (SHOW_PREFIX_2 and np.isnan(r_2)):
                print(f"{patient} excluded due to NaN values.")
                continue

            # Append to the respective list
            r_values_prefix_1.append(r_1)
            if SHOW_PREFIX_2:
                r_values_prefix_2.append(r_2)

            # Print the scores for this patient
            if SHOW_PREFIX_2:
                print(f"{patient}: {LABEL_1} = {r_1:.2f}, {LABEL_2} = {r_2:.2f}")
            else:
                print(f"{patient}: {LABEL_1} = {r_1:.2f}")

        except Exception as e:
            print(f"Error loading or processing data for patient {patient}, metric {metric}: {str(e)}")
            continue

    # Create the box and whisker plot
    data = [r_values_prefix_1]
    labels = [LABEL_1]
    
    if SHOW_PREFIX_2:
        data.append(r_values_prefix_2)
        labels.append(LABEL_2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False, labels=labels, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    # Capitalize the first letter of the metric
    metric_label = metric.capitalize()

    # Title with the metric and number of patients (updated to reflect the current count)
    plt.title(f'Predicting {metric_label}, N = {len(r_values_prefix_1)}', fontsize=24)

    # Set axis labels
    plt.xlabel("Pearson's R", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save the plot
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR.png'), bbox_inches='tight')
    plt.close()

print("Plots generated and saved successfully.")
