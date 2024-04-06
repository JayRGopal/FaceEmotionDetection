
def plot_au_presence_scatterplots(data_list):
    # Assuming data_list is a list of dictionaries in the order: Filtered, Reverse Filtered, Unfiltered
    
    conditions = ['Filtered', 'Reverse Filtered', 'Unfiltered']
    
    # Extract unique AUs and video filenames
    AUs = set()
    video_filenames = set()
    for data in data_list:
        for video_df_map in data.values():
            for video_filename, df in video_df_map.items():
                video_filenames.add(video_filename)
                AUs.update(df['AU'].unique())

    # Plotting
    for AU in AUs:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex='none', sharey='none')
        
        pres_pct_values = {condition: [] for condition in conditions}
        for condition, data in zip(conditions, data_list):
            for video_filename in sorted(video_filenames):
                df = data[next(iter(data))][video_filename]
                if df[df['AU'] == AU].empty:
                    pres_pct = 0
                else:
                    pres_pct = df[df['AU'] == AU]['pres_pct'].mean()
                pres_pct_values[condition].append(pres_pct)

        # Scatterplot 1: Reverse Filtered on y-axis, Filtered on x-axis
        r_value, p_value = pearsonr(pres_pct_values['Filtered'], pres_pct_values['Reverse Filtered'])
        sns.regplot(ax=axs[0], x=pres_pct_values['Filtered'], y=pres_pct_values['Reverse Filtered'], ci=None, label=f"Pearson's r: {r_value:.2f}, p: {p_value:.2e}")
        axs[0].set_title('Filtered vs Reverse Filtered AU Pres Pct')
        axs[0].set_xlabel('Filtered AU Pres Pct')
        axs[0].set_ylabel('Reverse Filtered AU Pres Pct')
        axs[0].legend()

        # Adjust y-axis limits based on Reverse Filtered data
        axs[0].set_ylim(bottom=min(pres_pct_values['Reverse Filtered']) - 5, top=max(pres_pct_values['Reverse Filtered']) + 5)

        # Scatterplot 2: Unfiltered on y-axis, Filtered on x-axis
        r_value, p_value = pearsonr(pres_pct_values['Filtered'], pres_pct_values['Unfiltered'])
        sns.regplot(ax=axs[1], x=pres_pct_values['Filtered'], y=pres_pct_values['Unfiltered'], ci=None, label=f"Pearson's r: {r_value:.2f}, p: {p_value:.2e}")
        axs[1].set_title('Filtered vs Unfiltered AU Pres Pct')
        axs[1].set_xlabel('Filtered AU Pres Pct')
        axs[1].set_ylabel('Unfiltered AU Pres Pct')
        axs[1].legend()

        # Adjust y-axis limits based on Unfiltered data
        axs[1].set_ylim(bottom=min(pres_pct_values['Unfiltered']) - 5, top=max(pres_pct_values['Unfiltered']) + 5)

        # Adjusting x-axis limits based on Filtered data
        for ax in axs:
            ax.set_xlim(left=min(pres_pct_values['Filtered']) - 5, right=max(pres_pct_values['Filtered']) + 5)

        plt.tight_layout()
        plt.show()
