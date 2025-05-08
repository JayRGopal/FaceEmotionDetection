import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Define file paths
timing_file = "/home/jgopal/NAS/SEEG-Smile-Events/S23_199_timing.csv"
au_file = "/home/jgopal/NAS/Analysis/outputs_EventAnalysis/combined_events_S23_199.csv"

# Read the data files
try:
    timing_df = pd.read_csv(timing_file)
    au_df = pd.read_csv(au_file)
    print(f"Successfully loaded data files")
    print(f"Timing data shape: {timing_df.shape}")
    print(f"AU data shape: {au_df.shape}")
except Exception as e:
    print(f"Error loading data files: {e}")

# Display the first few rows of each dataframe
print("\nTiming Data Preview:")
print(timing_df.head())

print("\nAU Data Preview:")
print(au_df.head())

# Filter timing data for Happiness and Neutral events only
timing_filtered = timing_df[timing_df['EventType'].isin(['Happiness', 'Neutral'])]
print(f"\nFiltered timing data shape: {timing_filtered.shape}")

# Get list of AU columns
au_columns = [col for col in au_df.columns if col.startswith('AU')]
print(f"\nDetected {len(au_columns)} AU columns: {au_columns}")

# Connect clip names between datasets
# Initialize dictionaries to store results
analysis_results = {
    "binarized_avg": {"smile": {}, "neutral": {}},
    "threshold_count": {"smile": {}, "neutral": {}},
    "raw_avg": {"smile": {}, "neutral": {}}
}

# Group the data by clip type (smile vs neutral)
smile_clips = timing_filtered[timing_filtered['EventType'] == 'Happiness']['ClipName'].tolist()
neutral_clips = timing_filtered[timing_filtered['EventType'] == 'Neutral']['ClipName'].tolist()

print(f"\nFound {len(smile_clips)} smile clips and {len(neutral_clips)} neutral clips")

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
            
            # 3. Raw averages
            
            # Store in results
            for au in au_columns:
                # Binarized average
                if au not in analysis_results["binarized_avg"][clip_type]:
                    analysis_results["binarized_avg"][clip_type][au] = []
                analysis_results["binarized_avg"][clip_type][au].append(clip_binarized[au])
                
                # Threshold count
                if au not in analysis_results["threshold_count"][clip_type]:
                    analysis_results["threshold_count"][clip_type][au] = []
                analysis_results["threshold_count"][clip_type][au].append(above_threshold[au])
                
                # Raw average
                if au not in analysis_results["raw_avg"][clip_type]:
                    analysis_results["raw_avg"][clip_type][au] = []
                analysis_results["raw_avg"][clip_type][au].append(clip_avg[au])
            
            # Collect all AU data for this clip type
            clip_au_data.append(clip_au_values)
        else:
            print(f"Warning: No AU data found for clip {clip}")
    
    # Combine all frame-level data for this clip type
    if clip_au_data:
        all_frames = pd.concat(clip_au_data)
        return all_frames
    return None

# Process all clips
smile_frames = process_clips(smile_clips, "smile")
neutral_frames = process_clips(neutral_clips, "neutral")

if smile_frames is not None and neutral_frames is not None:
    print(f"Collected {len(smile_frames)} frames for smile clips")
    print(f"Collected {len(neutral_frames)} frames for neutral clips")
else:
    print("Warning: One or both expression types have no data")

# Function to perform t-tests and create bar plots
def analyze_and_plot(data_type, title_suffix=""):
    smile_data = analysis_results[data_type]["smile"]
    neutral_data = analysis_results[data_type]["neutral"]
    
    # Prepare data for plotting
    mean_smile = {au: np.mean(smile_data[au]) for au in au_columns}
    mean_neutral = {au: np.mean(neutral_data[au]) for au in au_columns}
    
    sem_smile = {au: stats.sem(smile_data[au]) for au in au_columns}
    sem_neutral = {au: stats.sem(neutral_data[au]) for au in au_columns}
    
    # Perform t-tests
    pvalues = {}
    significant_smile = []
    significant_neutral = []
    
    for au in au_columns:
        t_stat, p_val = stats.ttest_ind(smile_data[au], neutral_data[au], equal_var=False)
        pvalues[au] = p_val
        
        if p_val < 0.05:
            if mean_smile[au] > mean_neutral[au]:
                significant_smile.append(au)
            else:
                significant_neutral.append(au)
    
    # Sort AUs by significance and difference magnitude
    sorted_aus = sorted(au_columns, 
                        key=lambda au: (pvalues[au] < 0.05, abs(mean_smile[au] - mean_neutral[au])),
                        reverse=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
        if pvalues[au] < 0.05:
            ax.text(i, max(mean_smile[au], mean_neutral[au]) + 
                   max(sem_smile[au], sem_neutral[au]) + 0.02, 
                   '*', ha='center', va='bottom', fontsize=16)
    
    # Customize plot
    ax.set_xlabel('Facial Action Units', fontsize=14, fontweight='bold')
    
    if data_type == "binarized_avg":
        ax.set_ylabel('Proportion of Clips Above Threshold (0.4)', fontsize=14, fontweight='bold')
        ax.set_title(f'Binarized AU Presence: Smile vs. Neutral{title_suffix}', fontsize=16, fontweight='bold')
    elif data_type == "threshold_count":
        ax.set_ylabel('Average Count of Frames Above Threshold (0.4)', fontsize=14, fontweight='bold')
        ax.set_title(f'Frame Count Above Threshold: Smile vs. Neutral{title_suffix}', fontsize=16, fontweight='bold')
    else:  # raw_avg
        ax.set_ylabel('Average AU Intensity (0-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'Raw AU Intensity: Smile vs. Neutral{title_suffix}', fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_aus, rotation=45, ha='right')
    ax.legend()
    
    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add a note about significance
    fig.text(0.5, 0.01, '* indicates p < 0.05 (significant difference)', ha='center', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print significant AUs
    print(f"\n--- Significant differences for {data_type} ---")
    print(f"AUs significantly higher in Smile: {', '.join(significant_smile)}")
    print(f"AUs significantly higher in Neutral: {', '.join(significant_neutral)}")
    
    return significant_smile, significant_neutral

# Perform all three analyses
print("\n=== ANALYSIS 1: BINARIZED AVERAGES ===")
sig_smile_bin, sig_neutral_bin = analyze_and_plot("binarized_avg")

print("\n=== ANALYSIS 2: THRESHOLD COUNT ===")
sig_smile_count, sig_neutral_count = analyze_and_plot("threshold_count")

print("\n=== ANALYSIS 3: RAW AVERAGES ===")
sig_smile_raw, sig_neutral_raw = analyze_and_plot("raw_avg")