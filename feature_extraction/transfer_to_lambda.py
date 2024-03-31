from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_au_presence_updated(data_dict, save_path):
    """
    Plots and saves a bar graph with more distinct colors for each AU's presence percentage across different videos.
    
    :param data_dict: Dictionary mapping video filenames to DataFrames with columns ['AU', 'pres_pct', 'int_mean', 'int_std']
    :param save_path: File path where the plot will be saved
    """
    # Figure setup
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    
    # Collecting all unique AUs
    all_aus = sorted(list(set(aus)))
    
    # Number of unique AUs and videos
    num_videos = len(data_dict)
    num_aus = len(all_aus)
    
    # Width of a bar
    width = 1 / (num_aus + 1)
    
    # Defining a colormap
    cmap = get_cmap('tab20', num_aus)  # Using a colormap with enough distinct colors
    
    # Creating a bar for each AU in each video with distinct colors
    for video_idx, (video_name, df) in enumerate(data_dict.items()):
        for au in all_aus:
            if au in df['AU'].values:
                pres_pct = df[df['AU'] == au]['pres_pct'].iloc[0]
                color = cmap(all_aus.index(au))
                ax.bar(video_idx + (all_aus.index(au) * width), pres_pct, width, color=color, label=au if video_idx == 0 else None)

    # Setting the x-axis labels
    ax.set_xticks(np.arange(num_videos) + width * (num_aus / 2))
    ax.set_xticklabels(video_filenames, rotation=45, ha="right")
    
    # Setting the y-axis label
    ax.set_ylabel('Presence Percentage')
    
    # Adding legend
    ax.legend(title="Action Units", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Title and grid
    plt.title('Presence Percentage of AUs across Videos')
    plt.grid(axis='y')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()