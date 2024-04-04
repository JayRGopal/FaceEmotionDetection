
def plot_au_presence(data_list):
    # Assuming data_list is a list of dictionaries in the order: Filtered, Reverse Filtered, Unfiltered
    
    conditions = ['Filtered', 'Reverse Filtered', 'Unfiltered']
    
    # Step 1: Extract unique AUs and video filenames
    AUs = set()
    video_filenames = set()
    for data in data_list:
        for video_df_map in data.values():
            for video_filename, df in video_df_map.items():
                video_filenames.add(video_filename)
                AUs.update(df['AU'].unique())
    
    # Step 2: Plotting
    for AU in AUs:
        fig, ax = plt.subplots(figsize=(len(video_filenames) * 1.5, 8))
        
        # Collecting pres_pct values for each condition and video for the current AU
        pres_pct_values = []
        for data in data_list:
            values_for_videos = []
            for video_filename in sorted(video_filenames):
                df = data[next(iter(data))][video_filename]
                pres_pct = df[df['AU'] == AU]['pres_pct'].mean()  # Average if there are multiple rows for the AU
                values_for_videos.append(pres_pct)
            pres_pct_values.append(values_for_videos)
        
        # Plotting
        x = range(len(video_filenames))
        width = 0.2  # Width of the bars
        for i, condition_values in enumerate(pres_pct_values):
            ax.bar([p + i*width for p in x], condition_values, width, label=conditions[i])
        
        ax.set_ylabel('AU Presence Percentage')
        ax.set_title(f'AU {AU} Presence Percentage by Video and Condition')
        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(sorted(video_filenames), rotation=45, ha='right')
        ax.legend(title='Condition')
        
        plt.tight_layout()
        plt.show()