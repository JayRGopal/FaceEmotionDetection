
def calculate_pearsons_r(features_dict, y):
    """
    Calculate Pearson's R correlation coefficient for each feature in each time window with the answers.

    Parameters:
    - features_dict: Dictionary with numerical keys for time_windows, each mapping to a (num_answers, num_features) array.
    - y: Numpy array of answers with shape (num_answers,).

    Returns:
    - A dictionary with the same time_window keys, where each key maps to a (num_features,) numpy array of Pearson's R values.
    """
    correlations = {}

    for time_window, features in features_dict.items():
        # Check if the dimensions match between features and answers
        if features.shape[0] != y.shape[0]:
            raise ValueError(f"Number of answers ({y.shape[0]}) does not match number of samples ({features.shape[0]}) in time window {time_window}.")

        # Initialize an array to store Pearson's R values for each feature
        pearsons_r_values = np.zeros(features.shape[1])

        # Calculate Pearson's R for each feature
        for feature_idx in range(features.shape[1]):
            feature_data = features[:, feature_idx]
            r, _ = pearsonr(feature_data, y)
            pearsons_r_values[feature_idx] = r

        correlations[time_window] = pearsons_r_values

    return correlations


def filter_features_by_correlation(features_dict, correlations, threshold):
    """
    Filter features based on Pearson's R correlation threshold.

    Parameters:
    - features_dict: Dictionary with numerical keys for time_windows, each mapping to a (num_answers, num_features) array.
    - correlations: Dictionary with the same time_window keys, mapping to (num_features,) numpy array of Pearson's R values.
    - threshold: Minimum Pearson's R correlation required to keep a feature.

    Returns:
    - A modified features_dict that only includes features with Pearson's R correlation >= threshold.
    """
    filtered_features_dict = {}

    for time_window, pearsons_r_values in correlations.items():
        # Identify features that meet or exceed the correlation threshold
        features_to_keep = np.where(pearsons_r_values >= threshold)[0]

        # Filter the original features based on the identified indices
        filtered_features = features_dict[time_window][:, features_to_keep]

        filtered_features_dict[time_window] = filtered_features

    return filtered_features_dict