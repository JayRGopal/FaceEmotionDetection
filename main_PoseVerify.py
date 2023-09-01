import os
import subprocess
import time
import datetime
from setup import download_file
import torch
import itertools
import json

from utilsMMPose import *

"""

Full Pipeline - MMPose

"""

# Set the parameters
VIDEO_DIRECTORY = os.path.abspath('MMPose_inputs/')
OUTPUT_DIRECTORY = os.path.abspath('outputs_MMPose_Verify/') # This is where json results go
OUTPUT_VIDEO_DIRECTORY = os.path.abspath('outputs_MMPose_Verify/') # This is where videos/images with overlay go 
TOP_DOWN = True
CONFIGS_BASE = os.path.abspath('mmpose/configs/body_2d_keypoint')
WHOLEBODY_CONFIGS_BASE = os.path.abspath('mmpose/configs/wholebody_2d_keypoint') 
MMPOSE_MODEL_BASE = os.path.abspath('MMPose_models/')
SUBJECT_FACE_IMAGE_PATH = os.path.abspath('deepface/Jimmy_Fallon.jpg') 
VERIFY_EVERY_FRAME = True
DEBUG = False # If debug is true, it stops verification & you can see all poses detected

# Model setup list
# (config_file, model_download, model_path, detector_setting)
# # FOR TOP DOWN WHOLE BODY
# model_setup_list = [
#   (f'{WHOLEBODY_CONFIGS_BASE}/rtmpose/coco-wholebody/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth', 'RTM'),
#   (f'{WHOLEBODY_CONFIGS_BASE}/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth', 'RTM'),
#   (f'{WHOLEBODY_CONFIGS_BASE}/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth', 'RTM')
# ]

# # FOR TOP DOWN BODY 2D
model_setup_list = [
  (f'{CONFIGS_BASE}/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py', 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth', f'{MMPOSE_MODEL_BASE}/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth', 'MM'),
  (f'{CONFIGS_BASE}/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py', 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth', f'{MMPOSE_MODEL_BASE}/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth', 'MM'),
  (f'{CONFIGS_BASE}/topdown_heatmap/coco/td-hm_hrformer-base_8xb32-210e_coco-384x288.py', 'https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_384x288-ecf0758d_20220316.pth', f'{MMPOSE_MODEL_BASE}/hrformer_base_coco_384x288-ecf0758d_20220316.pth', 'MM'),
  (f'{CONFIGS_BASE}/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py', 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth', f'{MMPOSE_MODEL_BASE}/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth', 'MM'),
  (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth', 'RTM'),
  (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-384x288.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth', 'RTM'),
  (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth', 'RTM')
]

# # FOR BOTTOM UP BODY 2D
# model_setup_list = [
#   (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth', 'RTM'),
#   (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-m_8xb256-420e_aic-coco-384x288.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-384x288-a62a0b32_20230228.pth', 'RTM'),
#   (f'{CONFIGS_BASE}/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-256x192.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth', f'{MMPOSE_MODEL_BASE}/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth', 'RTM')
# ]

# Detector mapping
# {detector_setting: (det_config_file, det_model_download, det_model_path)}
detector_mapping = {
  'RTM': ('mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', f'{MMPOSE_MODEL_BASE}/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'),
  'RTMN': ('mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py', 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth', f'{MMPOSE_MODEL_BASE}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'),
  'MM': ('mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', f'{MMPOSE_MODEL_BASE}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
}


# Parameter grid search
# {parameter flag}: list of values
parameter_search = {
  '--nms-thr': [0.2], 
  '--bbox-thr': [0.3],
  '--kpt-thr': [0.3]
}

# Get all combinations of parameters
parameter_combinations = list(itertools.product(*parameter_search.values()))

# Combine parameter_search and parameter_combinations into a new dictionary
combined_data = {
    'parameter_search': parameter_search,
    'parameter_combinations': parameter_combinations
}

# Save the combined data to a JSON file
with open(os.path.join(OUTPUT_DIRECTORY, 'parameter_combinations.json'), 'w') as file:
    json.dump(combined_data, file)

with open(os.path.join(OUTPUT_VIDEO_DIRECTORY, 'parameter_combinations.json'), 'w') as file:
    json.dump(combined_data, file)

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# For timing estimation
valid_videos = [vid for vid in all_videos if os.path.isfile(os.path.join(VIDEO_DIRECTORY, vid))]
num_vids = len(valid_videos)
start_time = time.time()

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for param_enum, combination in enumerate(parameter_combinations):
  # Create a dictionary with parameter names and values
  parameters = dict(zip(parameter_search.keys(), combination))
  parameter_string = ' '.join([f'{key} {value}' for key, value in parameters.items()])

  # Loop through all model setups
  for (config_file, model_download, model_path, det_setting) in model_setup_list:

    # Download model if not already there
    if not(os.path.exists(model_path)): 
      print(f'DOWNLOADING TO {model_path}')
      download_file(model_download, model_path)

    det_config_file, det_model_download, det_model_path = detector_mapping[det_setting]

    # Download detector model if not already there
    if not(os.path.exists(det_model_path)):
      print(f'DOWNLOADING TO {det_model_path}') 
      download_file(det_model_download, det_model_path) 

    
    # combine parameter combination number with model name to get folder for saving!
    model_base = f'{param_enum}_' + os.path.split(model_path)[-1]
    os.makedirs(os.path.join(OUTPUT_DIRECTORY, model_base), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_VIDEO_DIRECTORY, model_base), exist_ok=True)

    # save parameters to a file
    with open(os.path.join(OUTPUT_DIRECTORY, model_base, 'parameters.json'), 'w') as file:
      json.dump(parameters, file)
    
    with open(os.path.join(OUTPUT_VIDEO_DIRECTORY, model_base, 'parameters.json'), 'w') as file:
      json.dump(parameters, file)

    df_list = []

    # Loop through all videos (or images)
    for i in all_videos:
      save_file = os.path.join(OUTPUT_DIRECTORY, f'{model_base}', 'results_' + i[:-4] + '.json') 
      video_path = os.path.join(VIDEO_DIRECTORY, i)
      if os.path.exists(save_file):
        print(f'Skipping Video/Image {i}: Output File Already Exists!')
      elif os.path.isfile(video_path):
        if TOP_DOWN:
          
          cmd = f'python mmpose/JayGopal/run_topdown_with_verify.py \
            "{os.path.abspath(det_config_file)}" \
            "{os.path.abspath(det_model_path)}" \
            "{os.path.abspath(config_file)}" \
            "{os.path.abspath(model_path)}" \
            --input "{video_path}" \
            --draw-heatmap \
            --save-predictions \
            --output-root "{os.path.abspath(f"{OUTPUT_DIRECTORY}/{model_base}/")}" \
            --output-video "{os.path.abspath(f"{OUTPUT_VIDEO_DIRECTORY}/{model_base}/")}" \
            --device {device} \
            --target-face-path {SUBJECT_FACE_IMAGE_PATH} \
            {"--debug" if DEBUG else ""} \
            {"--verifyAll" if VERIFY_EVERY_FRAME else ""} \
            {parameter_string}' 
        else:
          cmd = f'python mmpose/JayGopal/run_bottomup.py \
            "{os.path.abspath(config_file)}" \
            "{os.path.abspath(model_path)}" \
            --input "{video_path}" \
            --output-root "{os.path.abspath(f"{OUTPUT_DIRECTORY}/{model_base}/")}" \
            --output-video "{os.path.abspath(f"{OUTPUT_VIDEO_DIRECTORY}/{model_base}/")}" \
            --save-predictions --draw-heatmap \
            --device {device}'

        subprocess.run(cmd, shell=True)
        if video_path[-4:] == '.mp4':
          df_temp = convert_to_df_vid(save_file)
        else:
          df_temp = convert_to_df(save_file)
        
        if len(df_temp.columns) > 1:
          df_temp.insert(0, 'Filename', [i]*len(df_temp))
          df_list.append(df_temp)
        
      else:
        print(f'WARNING: Got path {video_path}, which is not a valid video or image file!')
    if len(df_list) > 0:
      df_combined = pd.concat(df_list, ignore_index=True)
      df_combined.to_csv(os.path.join(OUTPUT_DIRECTORY, f'{model_base}/combined.csv'), index=False)
    # Time estimation
    elapsed_time = time.time() - start_time
    total_iterations = (len(model_setup_list)*len(parameter_combinations))
    iterations_left = total_iterations - (len(model_setup_list)*param_enum) - model_setup_list.index( (config_file, model_download, model_path, det_setting) ) - 1
    time_per_iteration = elapsed_time / (total_iterations - iterations_left) 
    time_left = time_per_iteration * iterations_left
    time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
    
    # print an update on the progress
    print('-' * 20)
    print('-' * 20)
    print("Approximately", time_left_formatted, "left to complete the operation")
    print('-' * 20)
    print('-' * 20)
  



  

  