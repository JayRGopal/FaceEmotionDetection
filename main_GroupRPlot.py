import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define the paths and prefixes
RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
PREFIX_1 = "OGAUHSE_L_"
PREFIX_2 = "OF_L_"

LABEL_1 = "FaceDx"
LABEL_2 = "OpenFace"

METRICS = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']  # Add all metrics you're interested in

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

# Detect all patients
patients = detect_patients()

# Loop through each metric
for metric in METRICS:
    r_values_prefix_1 = []
    r_values_prefix_2 = []

    # Loop through each patient
    for patient in patients:
        try:
            # Load the predictions for both prefixes
            predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1}_{metric}', RUNTIME_VAR_PATH)
            predictions_prefix_2 = load_var(f'predictions_{patient}_{PREFIX_2}_{metric}', RUNTIME_VAR_PATH)

            # Calculate Pearson's R for each
            r_1 = np.corrcoef(predictions_prefix_1['y_true'], predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']])[0, 1]
            r_2 = np.corrcoef(predictions_prefix_2['y_true'], predictions_prefix_2['preds'][predictions_prefix_2['best_time_radius']])[0, 1]

            # Append to the respective list
            r_values_prefix_1.append(r_1)
            r_values_prefix_2.append(r_2)

        except Exception as e:
            print(f"Error loading or processing data for patient {patient}, metric {metric}: {str(e)}")
            continue

    # Create the box and whisker plot
    data = [r_values_prefix_1, r_values_prefix_2]
    labels = [LABEL_1, LABEL_2]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False, labels=labels, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    # Capitalize the first letter of the metric
    metric_label = metric.capitalize()

    # Title with the metric and number of patients
    plt.title(f'Predicting {metric_label}, N = {len(patients)}', fontsize=24)

    # Set axis labels
    plt.xlabel("Pearson's R", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save the plot
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR.png'), bbox_inches='tight')
    plt.close()

print("Plots generated and saved successfully.")
