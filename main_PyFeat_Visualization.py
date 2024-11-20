import os
import numpy as np
from feat.plotting import plot_face
import matplotlib.pyplot as plt

# Set the patient ID and results path
PAT_SHORT_NAME = "S_199"  # Change this as needed
RESULTS_PATH_BASE = f"/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/{PAT_SHORT_NAME}/OGAU_L_/"
METRIC_NOW = 'Mood'

# Ensure the results folder exists
os.makedirs(RESULTS_PATH_BASE, exist_ok=True)

# Define AU activations manually (e.g., 1 = activated, 0 = not activated, or intensities 0-3)
# Ordered AUs are: 1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43
au_order = [1, 2, 4, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]
au_activations = {
    1: 3,  # AU1 activated at intensity 3
    2: 0,  # AU2 not activated
    12: 2, # AU12 activated at intensity 2 (e.g., smiling)
    43: 1, # AU43 activated at intensity 1 (e.g., eyes closed)
}

# Convert dictionary to numpy array with 20 elements
au_vector = np.zeros(len(au_order))
for au, intensity in au_activations.items():
    if au in au_order:
        index = au_order.index(au)
        au_vector[index] = intensity  # Map AU to the correct index


# Create muscle heatmap settings
muscles = {'all': 'heatmap'}

# Plot the face with muscle heatmaps
ax = plot_face(au=au_vector, muscles=muscles, title=f"Patient {PAT_SHORT_NAME} - {METRIC_NOW} AU Activation")

# Save the plot to the results folder
output_path = os.path.join(RESULTS_PATH_BASE, f"{METRIC_NOW}_face_heatmap.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[LOG] Face heatmap saved to {output_path}")
