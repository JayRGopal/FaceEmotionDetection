import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr

def plot_features_vs_labels(vectors_return, y, feature_names, save_path):
    """
    Generates scatter plots of features versus MADRS scores, including Pearson's R, p-value, and line of best fit.

    :param vectors_return: Dictionary containing one key and an array of shape (num_videos, num_features) as its value
    :param y: Array of shape (num_videos,) with ground truth labels for each video
    :param feature_names: List of strings with the names of each feature
    :param save_path: Path where the figure should be saved
    """
    # Extract the array of features from the dictionary
    features_array = next(iter(vectors_return.values()))
    
    # Determine the number of features
    num_features = features_array.shape[1]
    
    # Setup the figure for subplots
    fig, axs = plt.subplots(num_features, figsize=(10, 5 * num_features))
    
    # Iterate over each feature
    for i in range(num_features):
        # Scatter plot for the current feature
        axs[i].scatter(features_array[:, i], y)
        
        # Calculate Pearson's R and p-value
        r_value, p_value = pearsonr(features_array[:, i], y)
        
        # Add line of best fit
        m, b = np.polyfit(features_array[:, i], y, 1)
        axs[i].plot(features_array[:, i], m*features_array[:, i] + b, color="red")
        
        # Set titles and labels
        axs[i].set_title(f'MADRS vs. {feature_names[i]} (R={r_value:.2f}, p={p_value:.2g})')
        axs[i].set_ylabel('MADRS')
        axs[i].set_xlabel(feature_names[i])
    
    plt.tight_layout()
    
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

# Example usage of the function would be something like this:
# plot_features_vs_labels(vectors_return, y, feature_names, 'path/to/save/your/figure.png')
