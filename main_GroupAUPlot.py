import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# --- Configuration ---
RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
METRIC = 'Mood'

# The pickle files are assumed to have filenames that start with "topAUs_"
# and are stored in RUNTIME_VAR_PATH.
top_aus_dir = RUNTIME_VAR_PATH

# --- Load Top AUs ---
all_aus = []  # to collect AU labels across all files

# Iterate over files in the runtime variable directory
for fname in os.listdir(top_aus_dir):
    if fname.startswith("topAUs_") and fname.endswith(".pkl"):
        filepath = os.path.join(top_aus_dir, fname)
        with open(filepath, 'rb') as f:
            try:
                top_aus_dict = pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Could not load {fname}: {e}")
                continue
        # Check if the loaded dictionary contains results for our metric of interest
        if METRIC in top_aus_dict:
            top_list = top_aus_dict[METRIC]
            # Each item in top_list may be a tuple (label, coefficient) or just a label
            for item in top_list:
                if isinstance(item, tuple):
                    au_label = item[0]  # Extract the label
                else:
                    au_label = item
                all_aus.append(au_label)

if not all_aus:
    print(f"[LOG] No top AU data found for metric '{METRIC}' in {top_aus_dir}")
    exit(1)

# --- Analyze Concordance ---
au_counts = Counter(all_aus)
# Sort AUs by frequency (descending)
sorted_au = sorted(au_counts.items(), key=lambda x: x[1], reverse=True)
labels, frequencies = zip(*sorted_au)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies)
plt.xlabel('AU Feature', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title(f'Concordance of Top AUs for {METRIC} across Patients', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save the plot in RESULTS_PATH_BASE (inside a group folder)
plot_path = os.path.join(RESULTS_PATH_BASE, f"{METRIC}_top_aus_concordance.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"[LOG] Bar plot saved to {plot_path}")
plt.show()