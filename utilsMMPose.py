import json
import os
import cv2
import pandas as pd
import imageio
import glob


"""

MERGE CSVS INTO ONE DF


"""


def concatenate_csvs_in_folders(main_folder, pickle_path):
    all_dfs = []  # List to store all dataframes

    # Traverse through main_folder
    for subdir, dirs, files in os.walk(main_folder):
        # Skip if the folder ends with _labeled
        if not subdir.endswith("_labeled"):
            for file in files:
                if file.endswith(".csv") and "machinelabels" not in file:  # Only process csv files
                    # Don't process machinelabels.csv
                    csv_path = os.path.join(subdir, file)

                    # Load CSV file into a DataFrame
                    df = pd.read_csv(csv_path)
                    
                    # If DataFrame has less than 3 columns, skip it
                    if len(df.columns) < 3:
                        continue

                    # Modify the 3rd column, prepend folder name to the filenames
                    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: f'{os.path.basename(subdir)}/{x}')

                    # Append DataFrame to the list
                    all_dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove rows that contain headers
    # Assuming 'filename' is a part of the header row in the 3rd column
    concatenated_df = concatenated_df[concatenated_df.iloc[:, 2] != 'filename']

    # Save the dataframe to a pickle file
    concatenated_df.to_pickle(pickle_path)

    return concatenated_df





"""

MERGE IMAGES INTO VIDEO

"""

def merge_images_to_video(image_directory, output_video, order_file_path, development=True):
    """
    Merge images from a directory and its subdirectories (excluding those ending with "_labeled") into a video.

    Args:
        image_directory (str): Path to the directory containing the images.
        output_video (str): Output video file name and path.
        order_file_path (str): Path to JSON file to store order of images.
        development (boolean): If true, only takes the first 20 folders!
    """

    # Create a list to store the image file paths
    image_files = []

    # Iterate over the subdirectories and files in the image_directory
    # list all subfolders in main directory
    subfolders = [f.path for f in os.scandir(image_directory) if f.is_dir() and not(f.path.endswith('_labeled'))]

    # development only - first 20 folders
    if development:
        subfolders = subfolders[:20]
    
    for subfolder in subfolders:
        image_files_now = []
        for image_format in ["jpg", "png", "gif", "jpeg"]:
            image_files_now.extend(glob.glob(f'{subfolder}/*.{image_format}'))
        
        image_files = image_files + image_files_now 

    # Sort the image file paths
    image_files.sort()

    # Dump the order into a JSON file
    with open(order_file_path, 'w') as order_file:
        json.dump(image_files, order_file)

    # Create a list to store the image frames
    frames = []

    first_shape = imageio.imread(image_files[0]).shape
 
    # Iterate over the image file paths and append the frames
    for image_path in image_files:
        one_im = imageio.imread(image_path)
        if not(one_im.shape == first_shape):
            print(f'SHAPE MISMATCH! {one_im.shape} for {image_path}') 
        frames.append(imageio.imread(image_path))

    # Save the frames as a video
    imageio.mimsave(output_video, frames, fps=30)

    print(f"Video created: {output_video}")



def resize_images(image_folder):

    # CAUTION!! THIS SAVES OVER THE IMAGES! MAKE A COPY!
    

    # Iterate over the subdirectories and files in the image_directory
    for root, dirs, files in os.walk(image_folder):
        # Remove subdirectories ending with "_labeled"
        dirs[:] = [d for d in dirs if not d.endswith('_labeled')]

        # Iterate over the files in each subdirectory
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Get the image file path
                image_path = os.path.join(root, file)

                # resize it!
                if os.path.isfile(image_path):
                    img = cv2.imread(image_path)
                    resized_img = cv2.resize(img, (960, 540)) # These are the dimensions of our demo video!
                    cv2.imwrite(image_path, resized_img)





"""

VIDEO PROCESSING

"""


def create_dataframe_vid(label_dict, instance_info):
    columns_data_all = []
    frame_ids = []
    # loop through each frame
    for enum, i in enumerate(instance_info):
        ii_now = i['instances']
        # loop through each person in one frame
        for j in ii_now:
          # Initialize an empty dictionary to store the column data
          columns_data = {}

          frame_ids.append(enum)
          coordinates = j['keypoints']
          coordinates_conf = j['keypoint_scores']
          bbox_cords = j['bbox'][0]
          bbox_cords_conf = j['bbox_score']
          # Iterate over the label dictionary and coordinates list
          for index, label in label_dict.items():
              # Get the x and y coordinates for the current label index
              x, y = coordinates[int(index)]

              col_conf_value = coordinates_conf[int(index)]

              # Create the column names
              column_x = f'{label} x'
              column_y = f'{label} y'
              column_conf = f'{label} conf'

              # Store the x and y coordinates in the column data dictionary
              columns_data[column_x] = [x]
              columns_data[column_y] = [y]
              columns_data[column_conf] = [col_conf_value]
          
          for enumb, i in enumerate(bbox_cords):
              columns_data[f'bbox_cord_{enumb}'] = [i]
          
          columns_data['bbox_score'] = [bbox_cords_conf]

          # Create a pandas DataFrame from the column data dictionary
          temp_df = pd.DataFrame(columns_data)
          columns_data_all.append(temp_df)

    # combine into one df and insert frame number
    dataframe = pd.concat(columns_data_all, ignore_index=True)
    dataframe.insert(0, 'Frame Number', frame_ids)


    return dataframe


def convert_to_df_vid(json_file_path):
  # Assumes the JSON has results from ONE VIDEO

  # Open the JSON file and load its contents
  with open(json_file_path, 'r') as file:
      json_data = json.load(file)
  meta_data = json_data['meta_info']['keypoint_id2name']
  instance_info = json_data['instance_info']
  #return instance_info

  our_df = create_dataframe_vid(meta_data, instance_info)

  return our_df


"""

IMAGE PROCESSING

"""


def create_dataframe(label_dict, wrapper):
    # Initialize an empty dictionary to store the column data
    columns_data = {}

    coordinates = wrapper['keypoints']
    coordinates_conf = wrapper['keypoint_scores']
    bbox_cords = wrapper['bbox'][0]
    bbox_cords_conf = wrapper['bbox_score'] 

    # Iterate over the label dictionary and coordinates list
    for index, label in label_dict.items():
        # Get the x and y coordinates for the current label index
        x, y = coordinates[int(index)]

        col_conf_value = coordinates_conf[int(index)]

        # Create the column names
        column_x = f'{label} x'
        column_y = f'{label} y'
        conf_label = f'{label} conf'

        # Store the x and y coordinates in the column data dictionary
        columns_data[column_x] = [x]
        columns_data[column_y] = [y]
        columns_data[conf_label] = [col_conf_value]

    for enumb, i in enumerate(bbox_cords):
        columns_data[f'bbox_cord_{enumb}'] = [i]
          
    columns_data['bbox_score'] = [bbox_cords_conf]

    # Create a pandas DataFrame from the column data dictionary
    dataframe = pd.DataFrame(columns_data)

    return dataframe


def convert_to_df(json_file_path):
    # Assumes the JSON has results from ONE IMAGE

    # Open the JSON file and load its contents
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    meta_data = json_data['meta_info']['keypoint_id2name']
    instance_info = json_data['instance_info']
    
    df_list = []
    for i in instance_info:
        df_now = create_dataframe(meta_data, i)
        df_list.append(df_now)


    df_combined = pd.concat(df_list)

    return df_combined

