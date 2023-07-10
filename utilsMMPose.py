import json
import os
import cv2
import pandas as pd
import imageio


"""

MERGE IMAGES INTO VIDEO

"""

def merge_images_to_video(image_directory, output_video, order_file_path):
    """
    Merge images from a directory into a video.

    Args:
        image_directory (str): Path to the directory containing the images.
        output_video (str): Output video file name and path.
        order_file_path (str): Path to JSON file to store order of images.
    """

    # Get a list of image file names in the folder
    image_files = sorted(os.listdir(image_directory))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Dump the order into a JSON file
    with open(order_file_path, 'w') as order_file:
        json.dump(image_files, order_file)

    # Create a list to store the image frames
    frames = []

    # Iterate over the images and append them to the frames list
    for image in image_files:
        image_path = os.path.join(image_directory, image)
        frames.append(imageio.imread(image_path))

    # Save the frames as a video
    imageio.mimsave(output_video, frames, fps=30)

    print(f"Video created: {output_video}")

def resize_images(image_folder):
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        if os.path.isfile(image_path):
            img = cv2.imread(image_path)
            resized_img = cv2.resize(img, (960, 540)) # These are the dimensions of our demo video!
            cv2.imwrite(image_path, resized_img)


# def merge_images_to_video(folder_path, output_video_path, order_file_path):
#     # Get a list of image file names in the folder
#     image_files = sorted(os.listdir(folder_path))
#     image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

#     # Save the order of merged images to a JSON file
#     with open(order_file_path, 'w') as order_file:
#         json.dump(image_files, order_file)

#     # Initialize the video writer
#     frame = cv2.imread(os.path.join(folder_path, image_files[0]))
#     frame_height, frame_width, _ = frame.shape
#     video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

#     # Merge images into video
#     for image_file in image_files:
#         image_path = os.path.join(folder_path, image_file)
#         image = cv2.imread(image_path)

#         # Write the image to the video
#         video_writer.write(image)

#     # Release the video writer
#     video_writer.release()


"""

VIDEO PROCESSING

"""


def create_dataframe_vid(label_dict, instance_info):
    # Initialize an empty dictionary to store the column data
    columns_data = {}

    columns_data_all = []
    frame_ids = []
    # loop through each frame
    for enum, i in enumerate(instance_info):
        ii_now = i['instances']
        # loop through each person in one frame
        for j in ii_now:
          frame_ids.append(enum)
          coordinates = j['keypoints']
          # Iterate over the label dictionary and coordinates list
          for index, label in label_dict.items():
              # Get the x and y coordinates for the current label index
              x, y = coordinates[int(index)]

              # Create the column names
              column_x = f'{label} x'
              column_y = f'{label} y'

              # Store the x and y coordinates in the column data dictionary
              columns_data[column_x] = [x]
              columns_data[column_y] = [y]

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


def create_dataframe(label_dict, coordinates):
    # Initialize an empty dictionary to store the column data
    columns_data = {}

    # Iterate over the label dictionary and coordinates list
    for index, label in label_dict.items():
        # Get the x and y coordinates for the current label index
        x, y = coordinates[int(index)]

        # Create the column names
        column_x = f'{label} x'
        column_y = f'{label} y'

        # Store the x and y coordinates in the column data dictionary
        columns_data[column_x] = [x]
        columns_data[column_y] = [y]

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
        df_now = create_dataframe(meta_data, i['keypoints'])
        df_list.append(df_now)


    df_combined = pd.concat(df_list)

    return df_combined

