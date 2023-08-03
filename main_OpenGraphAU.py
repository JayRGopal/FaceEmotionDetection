from utilsHSE import *
from utils import *
import os
import time
import datetime
import cv2

"""

Full Pipeline - OpenGraphAU

"""


# Set the parameters
BATCH_SIZE = 50000
MODEL_TYPE = 'OpenGraphAU'
MODEL_BACKBONE = 'resnet50'
INPUT_SIZE = (224, 224)
POST_PROCESSING_METHOD = 'EMA'
VIDEO_DIRECTORY = os.path.abspath('inputs/')
FPS_EXTRACTING = 5 # we'll extract 5 fps

SAVE_PATH_FOLDER = lambda video_name: os.path.join(os.path.abspath('outputs_OpenGraphAU'), f'{video_name}')
SAVE_PATH = lambda save_path_folder, starter_frame: os.path.join(save_path_folder, f'{starter_frame}.csv')
SAVE_PATH_POST = lambda save_path_folder, starter_frame: os.path.join(save_path_folder, f'{starter_frame}_post.csv')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# For timing estimation
valid_videos = [vid for vid in all_videos if os.path.isfile(os.path.join(VIDEO_DIRECTORY, vid))]
unprocessed_videos = [vid for vid in valid_videos if not(os.path.exists(SAVE_PATH_FOLDER(vid)))]
num_vids = len(unprocessed_videos)
start_time = time.time()

TIMING_VERBOSE = True # yes/no do we print times for sub-processes within videos?



# Loop through all videos
for i in all_videos:
  # Process the entirety of each video via a while loop!

  video_path = os.path.join(VIDEO_DIRECTORY, i)

  if not(os.path.isfile(video_path)):
    # Case: Path isn't a file (usually happens if it's a folder)
    print(f'Not a valid path: {video_path}')
  else:
    # We know the path is to a file

    frame_now = 0 # this is what we save in outputs file
    frame_printing = 0 # this is the "real" frame we are at

    fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING) # FPS at which we're extracting

    save_path_folder = SAVE_PATH_FOLDER(i)

    if os.path.exists(save_path_folder):
      # Case: output folder already exists
      print(f'Skipping Video {i}: Output Folder Already Exists!')
    else:
      # We know the output folder does NOT exist already

      os.mkdir(save_path_folder)
      save_path_now = SAVE_PATH(save_path_folder, 0)
      save_path_now_post = SAVE_PATH_POST(save_path_folder, 0) 

      if TIMING_VERBOSE: 
        time1 = time.time()

      # Extract video frames
      capture = cv2.VideoCapture(video_path)
      ims = []
      real_fps = math.ceil(capture.get(cv2.CAP_PROP_FPS)) # real FPS of the video
      frame_division = real_fps // FPS_EXTRACTING # Helps us only analyze 5 fps (or close to it)
      running = True
      frameNr = 0 # Track frame number
      while running:
          # Extract frames continuously
          success, frame = capture.read()
          if success:
              if frameNr % frame_division == 0:
                  # We are only saving SOME frames (e.g. extracting 5 fps)
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  ims.append(frame)
              if (frameNr % BATCH_SIZE == 0) and (frameNr > 0):
                  # Let's do analysis, save results, and reset ims!
                  ims = np.array(ims)
                  print(f"Extracted Ims, Frames {frame_printing} to {frame_printing+BATCH_SIZE} in {i}") 
                  if TIMING_VERBOSE:
                    time2 = time.time()
                    print('Time: ', time2 - time1)
                  
                  # Batch now -- number of frames actually extracted (useful at end of video)
                  BATCH_NOW = ims.shape[0]

                  # Face detection
                  faces, is_null = extract_faces_mtcnn(ims, INPUT_SIZE)
                  faces = mtcnn_to_torch_new(faces)
                  print(f"Detected Faces")
                  if TIMING_VERBOSE:
                    time3 = time.time()
                    print('Time: ', time3 - time2) 
                  
                  # Get model predictions (OpenGraphAU)
                  net = load_network(model_type=MODEL_TYPE, backbone=MODEL_BACKBONE)
                  predictions = get_model_preds(faces, net, model_type=MODEL_TYPE)
                  predictions[is_null == 1] = 0 # clear the predictions from frames w/o faces!
                  print("Got Network Predictions")
                  if TIMING_VERBOSE:
                    time4 = time.time()
                    print('Time: ', time4 - time3)

                  # Post-processing
                  preds_post = postprocess_outs(predictions, method=POST_PROCESSING_METHOD)
                  # TODO: How do we deal will null frames in post-processing?
                  print("Post Processing Complete")

                  # Save outputs to a CSV
                  frames = np.arange(frame_now, frame_now + BATCH_NOW).reshape(BATCH_NOW, 1)
                  csv_save(labels=predictions, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
                  print(f"Saved Raw Predictions to {save_path_now}!")
                  csv_save(labels=preds_post, is_null=is_null, frames=frames, save_path=save_path_now_post, fps=fps)
                  print(f"Saved Post-Processed to {save_path_now_post}!")

                  frame_now = frame_now + BATCH_NOW
                  frame_printing = frame_printing + BATCH_SIZE 

                  # Reset ims for the next batch!
                  ims = []

                  # Reset timing
                  if TIMING_VERBOSE: 
                    time1 = time.time()
          else:
              # We're out of frames!
              running = False

              # Let's do analysis, save results, and reset ims!
              ims = np.array(ims)
              print(f"Extracted Ims, Frames {frame_printing} to {frame_printing+BATCH_SIZE} in {i}") 
              if TIMING_VERBOSE:
                time2 = time.time()
                print('Time: ', time2 - time1)
              
              # Batch now -- number of frames actually extracted (useful at end of video)
              BATCH_NOW = ims.shape[0]

              # Face detection
              faces, is_null = extract_faces_mtcnn(ims, INPUT_SIZE)
              faces = mtcnn_to_torch_new(faces)
              print(f"Detected Faces")
              if TIMING_VERBOSE:
                time3 = time.time()
                print('Time: ', time3 - time2) 
              
              # Get model predictions (OpenGraphAU)
              net = load_network(model_type=MODEL_TYPE, backbone=MODEL_BACKBONE)
              predictions = get_model_preds(faces, net, model_type=MODEL_TYPE)
              predictions[is_null == 1] = 0 # clear the predictions from frames w/o faces!
              print("Got Network Predictions")
              if TIMING_VERBOSE:
                time4 = time.time()
                print('Time: ', time4 - time3)

              # Post-processing
              preds_post = postprocess_outs(predictions, method=POST_PROCESSING_METHOD)
              # TODO: How do we deal will null frames in post-processing?
              print("Post Processing Complete")

              # Save outputs to a CSV
              frames = np.arange(frame_now, frame_now + BATCH_NOW).reshape(BATCH_NOW, 1)
              csv_save(labels=predictions, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
              print(f"Saved Raw Predictions to {save_path_now}!")
              csv_save(labels=preds_post, is_null=is_null, frames=frames, save_path=save_path_now_post, fps=fps)
              print(f"Saved Post-Processed to {save_path_now_post}!")

              frame_now = frame_now + BATCH_NOW
              frame_printing = frame_printing + BATCH_SIZE 

              # Reset ims to save room!
              ims = []

              # Reset timing
              if TIMING_VERBOSE: 
                time1 = time.time()

          frameNr = frameNr + 1
      capture.release()

      # Time estimation
      elapsed_time = time.time() - start_time
      iterations_left = num_vids - unprocessed_videos.index(i) - 1
      time_per_iteration = elapsed_time / (unprocessed_videos.index(i) + 1)
      time_left = time_per_iteration * iterations_left
      time_left_formatted = str(datetime.timedelta(seconds=int(time_left)))
      
      # print an update on the progress
      print("Approximately ", time_left_formatted, " left to complete analyzing all videos")

 
