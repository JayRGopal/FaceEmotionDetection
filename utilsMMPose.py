import json

import pandas as pd


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

def create_dataframe(label_dict, instance_info):
    # Initialize an empty dictionary to store the column data
    columns_data = {}

    num_people = len(instance_info)
    print(num_people) 
    if num_people == 1:
      coordinates = instance_info[0]['keypoints']
    
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

  our_df = create_dataframe(meta_data, instance_info)

  return our_df

