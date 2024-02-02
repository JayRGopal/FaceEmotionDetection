
def calculate_average_rows(df_dict, ground_truth_df):
    # Initialize counters and row totals
    low_pain_count, high_pain_count = 0, 0
    low_pain_rows_total, high_pain_rows_total = 0, 0
    
    # Convert keys in df_dict to the same timestamp format as in ground_truth_df if necessary
    converted_df_dict = {pd.to_datetime(key, errors='coerce', format='mixed'): value for key, value in df_dict.items()}
    
    # Iterate through the ground truth DataFrame
    for _, row in ground_truth_df.iterrows():
        timestamp = pd.to_datetime(row['Datetime'], errors='coerce', format='mixed')  # Ensure timestamp format matches
        pain_level = row['Pain']
        
        # Check if the timestamp exists in the converted dictionary
        if timestamp in converted_df_dict:
            df = converted_df_dict[timestamp]
            num_rows = len(df)
            
            # Accumulate totals and counts based on pain level
            if pain_level == 0:  # Low pain
                low_pain_rows_total += num_rows
                low_pain_count += 1
            elif pain_level == 1:  # High pain
                high_pain_rows_total += num_rows
                high_pain_count += 1
                
    # Calculate averages, avoiding division by zero
    avg_low_pain = low_pain_rows_total / low_pain_count if low_pain_count else 0
    avg_high_pain = high_pain_rows_total / high_pain_count if high_pain_count else 0
    
    return (avg_low_pain, avg_high_pain)


def plot_average_rows_changes(time_window_dict):
    # Lists to store the data
    time_windows = []
    avg_low_pain = []
    avg_high_pain = []
    
    # Process the dictionary
    for window, (low, high) in time_window_dict.items():
        time_windows.append(window)
        avg_low_pain.append(low)
        avg_high_pain.append(high)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_windows, avg_low_pain, label='Low Pain', marker='o')
    plt.plot(time_windows, avg_high_pain, label='High Pain', marker='x')
    
    plt.title('Average Number of Rows Across Time Windows')
    plt.xlabel('Time Window')
    plt.ylabel('Average Number of Rows')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
