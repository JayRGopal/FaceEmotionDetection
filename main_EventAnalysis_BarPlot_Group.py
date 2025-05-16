import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Define base directories
base_timing_dir = "/home/jgopal/NAS/SEEG-Smile-Events/"
base_au_dir = "/home/jgopal/NAS/Analysis/outputs_EventAnalysis/"
output_dir = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_May_2025/'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Find all patient timing files
timing_files = glob.glob(f"{base_timing_dir}/*_timing.csv")
patient_codes = [os.path.basename(f).split('_timing')[0] for f in timing_files]

print(f"Found {len(patient_codes)} patients: {patient_codes}")

# Initialize dictionaries to store aggregated results across all patients
all_patients_results = {
    "binarized_avg": {"smile": {}, "neutral": {}},
    "threshold_count": {"smile": {}, "neutral": {}},
    "raw_avg": {"smile": {}, "neutral": {}}
}

# Process each patient
for patient_code in patient_codes:
    print(f"\n===== Processing patient: {patient_code} =====")
    
    # Define file paths for this patient
    timing_file = f"{base_timing_dir}/{patient_code}_timing.csv"
    au_file = f"{base_au_dir}/combined_events_{patient_code}.csv"
    
    # Check if both files exist
    if not os.path.exists(timing_file) or not os.path.exists(au_file):
        print(f"Skipping patient {patient_code} - missing files")
        continue
    
    # Read the data files
    try:
        timing_df = pd.read_csv(timing_file)
        au_df = pd.read_csv(au_file)
        print(f"Successfully loaded data files")
        print(f"Timing data shape: {timing_df.shape}")
        print(f"AU data shape: {au_df.shape}")
    except Exception as e:
        print(f"Error loading data files for {patient_code}: {e}")
        continue
    
    # Filter timing data for Happiness and Neutral events only
    timing_filtered = timing_df[timing_df['EventType'].isin(['Happiness', 'Neutral'])]
    print(f"Filtered timing data shape: {timing_filtered.shape}")
    
    # Get list of AU columns (only include those that start with 'AU' but not 'AUL' or 'AUR')
    au_columns = [col for col in au_df.columns if col.startswith('AU') and not col.startswith(('AUL', 'AUR'))]
    
    if not au_columns:
        print(f"No AU columns found for patient {patient_code}")
        continue
        
    print(f"Detected {len(au_columns)} AU columns")
    
    # Initialize patient-specific results
    patient_results = {
        "binarized_avg": {"smile": {}, "neutral": {}},
        "threshold_count": {"smile": {}, "neutral": {}},
        "raw_avg": {"smile": {}, "neutral": {}}
    }
    
    # Group the data by clip type (smile vs neutral)
    smile_clips = timing_filtered[timing_filtered['EventType'] == 'Happiness']['ClipName'].tolist()
    neutral_clips = timing_filtered[timing_filtered['EventType'] == 'Neutral']['ClipName'].tolist()
    
    print(f"Found {len(smile_clips)} smile clips and {len(neutral_clips)} neutral clips")
    
    # Function to process clips of a specific type
    def process_clips(clip_list, clip_type):
        clip_au_data = []
        
        for clip in clip_list:
            # Find corresponding rows in AU dataframe
            clip_data = au_df[au_df['Clip Name'] == clip]
            
            if len(clip_data) > 0:
                # Extract AU values for this clip
                clip_au_values = clip_data[au_columns]
                
                # Calculate different metrics
                # 1. Average then binarize
                clip_avg = clip_au_values.mean()
                clip_binarized = (clip_avg >= 0.4).astype(int)
                
                # 2. Count frames above threshold
                above_threshold = (clip_au_values >= 0.4).sum()
                
                # Store in results
                for au in au_columns:
                    # Binarized average
                    if au not in patient_results["binarized_avg"][clip_type]:
                        patient_results["binarized_avg"][clip_type][au] = []
                    patient_results["binarized_avg"][clip_type][au].append(clip_binarized[au])
                    
                    # Threshold count
                    if au not in patient_results["threshold_count"][clip_type]:
                        patient_results["threshold_count"][clip_type][au] = []
                    patient_results["threshold_count"][clip_type][au].append(above_threshold[au])
                    
                    # Raw average
                    if au not in patient_results["raw_avg"][clip_type]:
                        patient_results["raw_avg"][clip_type][au] = []
                    patient_results["raw_avg"][clip_type][au].append(clip_avg[au])
                
                # Collect all AU data for this clip type
                clip_au_data.append(clip_au_values)
            else:
                print(f"Warning: No AU data found for clip {clip}")
        
        # Combine all frame-level data for this clip type
        if clip_au_data:
            all_frames = pd.concat(clip_au_data)
            return all_frames
        return None
    
    # Process all clips for this patient
    smile_frames = process_clips(smile_clips, "smile")
    neutral_frames = process_clips(neutral_clips, "neutral")
    
    if smile_frames is not None and neutral_frames is not None:
        print(f"Collected {len(smile_frames)} frames for smile clips")
        print(f"Collected {len(neutral_frames)} frames for neutral clips")
    else:
        print(f"Warning: One or both expression types have no data for patient {patient_code}")
        continue
    
    # Add patient results to aggregated results
    for data_type in ["binarized_avg", "threshold_count", "raw_avg"]:
        for expression in ["smile", "neutral"]:
            for au in au_columns:
                if au not in patient_results[data_type][expression]:
                    continue
                    
                # Initialize if this AU hasn't been seen before
                if au not in all_patients_results[data_type][expression]:
                    all_patients_results[data_type][expression][au] = []
                
                # Add this patient's average for this AU to the aggregated results
                au_value = np.mean(patient_results[data_type][expression][au])
                all_patients_results[data_type][expression][au].append(au_value)

# Create a list of all unique AUs across all patients
all_aus = set()
for data_type in ["binarized_avg"]:  # Only focus on binarized_avg as requested
    for expression in ["smile", "neutral"]:
        all_aus.update(all_patients_results[data_type][expression].keys())

all_aus = sorted(list(all_aus))
print(f"\nFound {len(all_aus)} unique Action Units across all patients")

# Function to perform t-tests and create aggregated bar plot
def analyze_and_plot_aggregated(data_type="binarized_avg"):
    smile_data = all_patients_results[data_type]["smile"]
    neutral_data = all_patients_results[data_type]["neutral"]
    
    # Check if we have data to analyze
    if not smile_data or not neutral_data:
        print("No data to analyze. Exiting.")
        return
    
    # Get common AUs that have data for both smile and neutral
    common_aus = [au for au in all_aus if au in smile_data and au in neutral_data]
    
    if not common_aus:
        print("No common AUs found between smile and neutral conditions.")
        return
        
    print(f"Analyzing {len(common_aus)} common AUs")
    
    # Prepare data for plotting
    mean_smile = {au: np.mean(smile_data[au]) for au in common_aus}
    mean_neutral = {au: np.mean(neutral_data[au]) for au in common_aus}
    
    sem_smile = {au: stats.sem(smile_data[au]) for au in common_aus}
    sem_neutral = {au: stats.sem(neutral_data[au]) for au in common_aus}
    
    # Perform t-tests
    pvalues = {}
    significant_levels = {}
    significant_smile = []
    significant_neutral = []
    
    for au in common_aus:
        # Some patients might not have data for all AUs
        if len(smile_data[au]) < 2 or len(neutral_data[au]) < 2:
            print(f"Skipping {au} due to insufficient data")
            pvalues[au] = 1.0
            significant_levels[au] = ''
            continue
            
        t_stat, p_val = stats.ttest_ind(smile_data[au], neutral_data[au], equal_var=False)
        pvalues[au] = p_val
        
        # Determine significance level
        if p_val < 0.05:
            significant_levels[au] = '*'
        else:
            significant_levels[au] = ''
        
        if p_val < 0.05:
            if mean_smile[au] > mean_neutral[au]:
                significant_smile.append(au)
            else:
                significant_neutral.append(au)
    
    # Sort AUs by significance and difference magnitude
    sorted_aus = sorted(common_aus, 
                      key=lambda au: (pvalues[au] < 0.05, abs(mean_smile[au] - mean_neutral[au])),
                      reverse=True)
    
    # --- Save all the bar plot data and metadata to CSV for external plotting ---
    export_rows = []
    for i, au in enumerate(sorted_aus):
        export_rows.append({
            "AU": au,
            "Smile_Mean": mean_smile[au],
            "Smile_SEM": sem_smile[au],
            "Neutral_Mean": mean_neutral[au],
            "Neutral_SEM": sem_neutral[au],
            "p_value": pvalues[au],
            "Significant": significant_levels[au],
            "Significant_Higher": (
                "Smile" if au in significant_smile else ("Neutral" if au in significant_neutral else "")
            ),
            "Bar_Index": i
        })
    # Add plot-level metadata as a separate row (or could be a separate file, but here as extra columns)
    plot_metadata = {
        "Plot_Title": f"Facial Action Unit Presence: Smile vs. Neutral (Aggregated Across {len(patient_codes)} Patients)",
        "X_Label": "Facial Action Units",
        "Y_Label": "Proportion of Clips (Averaged Across Patients)" if data_type == "binarized_avg" else "",
        "Legend_Smile": "Smile",
        "Legend_Neutral": "Neutral",
        "Legend_Significance": "p < 0.05",
        "Num_Patients": len(patient_codes),
        "Data_Type": data_type
    }
    # Add metadata to every row for convenience
    for row in export_rows:
        row.update(plot_metadata)
    export_df = pd.DataFrame(export_rows)
    csv_filename = f"{output_dir}/AU_binarized_presence_ALL_PATIENTS_plotdata.csv"
    export_df.to_csv(csv_filename, index=False)
    print(f"Bar plot data and metadata saved to: {csv_filename}")
    # --- End CSV export ---

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(sorted_aus))
    width = 0.35
    
    # Plot bars
    smile_bars = ax.bar(x - width/2, [mean_smile[au] for au in sorted_aus], width, 
                       label='Smile', color='#FFA500', 
                       yerr=[sem_smile[au] for au in sorted_aus], capsize=5)
    
    neutral_bars = ax.bar(x + width/2, [mean_neutral[au] for au in sorted_aus], width,
                        label='Neutral', color='#4682B4', 
                        yerr=[sem_neutral[au] for au in sorted_aus], capsize=5)
    
    # Add significance markers
    for i, au in enumerate(sorted_aus):
        if significant_levels[au]:
            height = max(mean_smile[au], mean_neutral[au]) + max(sem_smile[au], sem_neutral[au]) + 0.02
            ax.text(i, height, significant_levels[au], ha='center', va='bottom', fontsize=14)
    
    # Customize plot
    ax.set_xlabel('Facial Action Units', fontsize=14, fontweight='bold')
    
    if data_type == "binarized_avg":
        ax.set_ylabel('Proportion of Clips (Averaged Across Patients)', fontsize=14, fontweight='bold')
        ax.set_title(f'Facial Action Unit Presence: Smile vs. Neutral\nAggregated Across {len(patient_codes)} Patients', 
                   fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_aus, rotation=45, ha='right')
    
    # Create custom legend with significance levels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FFA500', lw=4, label='Smile'),
        Line2D([0], [0], color='#4682B4', lw=4, label='Neutral'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='p < 0.05', linestyle='None')
    ]
    first_legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add the first legend
    ax.add_artist(first_legend)
    
    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add padding to avoid text overlap
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    plot_filename = f"{output_dir}/AU_binarized_presence_ALL_PATIENTS.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Print significant AUs
    print(f"\n--- Significant differences for {data_type} across all patients ---")
    print(f"AUs significantly higher in Smile: {', '.join(significant_smile)}")
    print(f"AUs significantly higher in Neutral: {', '.join(significant_neutral)}")
    
    return significant_smile, significant_neutral

# Perform analysis for binarized averages across all patients
print(f"\n=== AGGREGATED ANALYSIS: BINARIZED AVERAGES ACROSS ALL PATIENTS ===")
sig_smile_bin, sig_neutral_bin = analyze_and_plot_aggregated("binarized_avg")