"""
This script loads saved pickle files containing top AU selections from different patients/settings,
filters for the "Mood" metric and for features with a positive model coefficient,
and then:
  1. Generates a bar plot showing the frequency (concordance) of these AU selections.
  2. Constructs a 20-dimensional AU intensity vector (using the standard AU order)
     and highlights the top 3 positive AUs by setting their intensity,
     then plots a facial expression using plot_face.
     
"""

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
# AUs: 1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43
au_order = [1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]

# --- Load Top AUs ---
# We'll collect only positive AU entries.
# Each entry is expected to be either a tuple (label, coefficient) or a plain label.
positive_aus = []  # list of AU labels (e.g., "AU12 Lip Corner Puller")
# Optionally, we could also track details (e.g., coefficients) if needed.
positive_details = []

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
                    if coeff > 0:  # only include positive coefficients
                        positive_aus.append(label)
                        positive_details.append((label, coeff))
                else:
                    # If not a tuple, we assume it to be a label. (Optionally, skip or include.)
                    positive_aus.append(item)
                    positive_details.append((item, None))

if not positive_aus:
    print(f"[LOG] No positive top AU data found for metric '{METRIC}' in {top_aus_dir}")
    exit(1)

# --- Analyze Concordance ---
au_counts = Counter(positive_aus)
# Sort AUs by frequency in descending order
sorted_aus = sorted(au_counts.items(), key=lambda x: x[1], reverse=True)
labels, frequencies = zip(*sorted_aus)

# --- Plot Bar Chart of AU Frequencies ---
plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies)
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

# --- Select Top 3 AUs and Plot Facial Expression ---
# Choose the top 3 AU labels from the sorted frequency list.
top3_labels = [label for label, count in sorted_aus[:3]]
print(f"[LOG] Top 3 positive AUs for {METRIC}: {top3_labels}")

# Create a neutral AU vector of length 19 (one value per AU in our order)
# (Indices correspond to: 1,2,4,...,43)
au_vector = np.zeros(len(au_order))

# For each top AU, extract its number and set a fixed intensity.
# We assume labels start with "AU" followed by the AU number.
for label in top3_labels:
    try:
        # Example label: "AU12 Lip Corner Puller ..." -> extract "12"
        parts = label.split()
        if parts[0].startswith("AU"):
            au_number = int(parts[0][2:])  # convert after "AU" to int
            if au_number in au_order:
                index = au_order.index(au_number)
                au_vector[index] = 5.0  # set a fixed intensity (adjust as desired)
            else:
                print(f"[WARNING] AU number {au_number} from label '{label}' not found in standard order.")
        else:
            print(f"[WARNING] Label '{label}' does not start with 'AU'.")
    except Exception as e:
        print(f"[ERROR] Could not parse label '{label}': {e}")

# Plot the face using the provided function
ax = plot_face(au=au_vector, title=f"Top 3 Positive AUs for {METRIC}")
faceplot_path = os.path.join(RESULTS_PATH_BASE, f"{METRIC}_top3_faceplot.png")
plt.savefig(faceplot_path, bbox_inches='tight')
print(f"[LOG] Face plot saved to {faceplot_path}")
plt.show()
