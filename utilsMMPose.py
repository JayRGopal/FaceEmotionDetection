import json

import pandas as pd

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
  keypoints = json_data['instance_info'][0]['keypoints']

  our_df = create_dataframe(meta_data, keypoints)

  return our_df

