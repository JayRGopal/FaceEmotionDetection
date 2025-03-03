import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Import the face plotting function (assumed available)
from feat.plotting import plot_face

# --- Configuration ---
RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
METRIC = 'Mood'
top_aus_dir = RUNTIME_VAR_PATH

# Standard AU order for the 20 AUs (ordered as in your example)
# AUs: 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43
au_order = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]

# --- Load Top AUs ---
# Separate positive and negative AU entries.
positive_aus = []  # list of AU labels with positive coefficients
negative_aus = []  # list of AU labels with negative coefficients

for fname in os.listdir(top_aus_dir):
    if fname.startswith("topAUs_") and fname.endswith(".pkl"):
        filepath = os.path.join(top_aus_dir, fname)
        try:
            with open(filepath, 'rb') as f:
                top_aus_dict = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Could not load {fname}: {e}")
            continue

        if METRIC in top_aus_dict:
            top_list = top_aus_dict[METRIC]
            for item in top_list:
                # If the entry is a tuple, expect (label, coefficient)
                if isinstance(item, tuple):
                    label, coeff = item
                    if coeff > 0:
                        positive_aus.append(label)
                    elif coeff < 0:
                        negative_aus.append(label)
                else:
                    # If not a tuple, include in positive by default (or skip)
                    positive_aus.append(item)

if not positive_aus:
    print(f"[LOG] No positive top AU data found for metric '{METRIC}' in {top_aus_dir}")
    exit(1)

# --- Analyze Concordance for Positive AUs ---
pos_counts = Counter(positive_aus)
sorted_pos_aus = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
pos_labels, pos_frequencies = zip(*sorted_pos_aus)

# --- Plot Bar Chart of Positive AU Frequencies ---
plt.figure(figsize=(10, 6))
plt.bar(pos_labels, pos_frequencies)
plt.xlabel('AU Feature', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title(f'Concordance of Positive Top AUs for {METRIC} across Patients', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

barplot_path = os.path.join(RESULTS_PATH_BASE, f"{METRIC}_top_aus_concordance.png")
plt.savefig(barplot_path, bbox_inches='tight')
print(f"[LOG] Bar plot saved to {barplot_path}")
plt.show()

# --- Prepare Facial Expression Plots ---
# Function to create and save a face plot given a list of AU labels
def create_face_plot(au_labels, direction, top_k):
    # Start with a neutral vector (all zeros) for 20 AUs
    au_vector = np.zeros(len(au_order))
    for label in au_labels:
        try:
            parts = label.split()
            if parts[0].startswith("AU"):
                au_number = int(parts[0][2:])  # extract number after "AU"
                if au_number in au_order:
                    idx = au_order.index(au_number)
                    # For both directions, set a fixed intensity (e.g., 5.0)
                    au_vector[idx] = 5.0
                else:
                    print(f"[WARNING] AU number {au_number} from label '{label}' not in standard order.")
            else:
                print(f"[WARNING] Label '{label}' does not start with 'AU'.")
        except Exception as e:
            print(f"[ERROR] Could not parse label '{label}': {e}")
    
    title = f"Top {top_k} {direction.capitalize()} AUs for {METRIC}"
    ax = plot_face(au=au_vector, title=title)
    # Save the plot
    filename = os.path.join(RESULTS_PATH_BASE, f"{METRIC}_top{top_k}_{direction}_faceplot.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"[LOG] Face plot saved to {filename}")
    plt.close()

# --- Plot Faces for Positive AUs ---
# We plot cumulative faces: top 1, top 2, ... top 5 positives.
for k in range(1, 6):
    # Get top k positive labels based on frequency
    top_k_pos = [label for label, count in sorted_pos_aus[:k]]
    create_face_plot(top_k_pos, direction="positive", top_k=k)

# --- Plot Faces for Negative AUs (if available) ---
if negative_aus:
    neg_counts = Counter(negative_aus)
    sorted_neg_aus = sorted(neg_counts.items(), key=lambda x: x[1], reverse=True)
    for k in range(1, 6):
        top_k_neg = [label for label, count in sorted_neg_aus[:k]]
        create_face_plot(top_k_neg, direction="negative", top_k=k)
else:
    print(f"[LOG] No negative top AU data found for metric '{METRIC}'.")
