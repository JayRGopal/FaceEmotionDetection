
def plot_features_vs_labels(vectors_return, y, feature_names, save_path):
    """
    Generates scatter plots of features versus MADRS scores with Pearson's R, p-value,
    and line of best fit using seaborn's regplot for enhanced visualization.

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
    
    if num_features == 1:
        axs = [axs]  # Make it iterable if there's only one subplot
    
    # Iterate over each feature
    for i, ax in enumerate(axs):
        # Use Seaborn's regplot for plotting scatter with regression line and confidence interval
        sns.regplot(x=features_array[:, i], y=y, ax=ax, ci=95, color="b", line_kws={'color': 'red'})
        
        # Calculate Pearson's R and p-value
        r_value, p_value = pearsonr(features_array[:, i], y)
        
        # Set titles and labels
        ax.set_title(f'MADRS vs. {feature_names[i]} (R={r_value:.2f}, p={p_value:.2g})')
        ax.set_ylabel('MADRS')
        ax.set_xlabel(feature_names[i])
    
    plt.tight_layout()
    
    # Check if the directory exists, and create it if it doesn't
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
