from utils import *
from utilsHSE import *
import os
import time
import datetime
import cv2



import torch
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)

from hsemotion.facial_emotions import HSEmotionRecognizer

def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    bounding_boxes=bounding_boxes[probs>0.9]
    return bounding_boxes

fer=HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf',device=device)

def hse_preds_new(faces, model, model_type='mobilenet_7.h5'):
    #import pdb; pdb.set_trace()
    # faces = preprocessing_function(faces)
    # scores=model.predict(faces)

    # Check if a GPU is available and use it if possible
    device_name = tf.test.gpu_device_name()
    if device_name != '' and '/device:GPU' in device_name:
        with tf.device('/device:GPU:0'):
            scores = model.predict(faces)
            scores = scores.cpu()
    else:
        scores = model.predict(faces)

    return scores


"""

Full Pipeline - HSEmotion

"""




# Set the parameters
BATCH_SIZE = 50000
MODEL_TYPE = 'mobilenet_7.h5'
INPUT_SIZE = (224, 224)
VIDEO_DIRECTORY = os.path.abspath('inputs/')
FPS_EXTRACTING = 5 # we'll extract 5 fps


SAVE_PATH_FOLDER = lambda video_name: os.path.join(os.path.abspath('outputs_HSEmotion'), f'{video_name}')
SAVE_PATH = lambda save_path_folder, starter_frame: os.path.join(save_path_folder, f'{starter_frame}.csv')

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

                  # --- DEBUG ---
                  for one_im in ims:
                    bounding_boxes=detect_face(one_im)
                    print('testing in one_im')
                    for bbox in bounding_boxes:
                      box = bbox.astype(int)
                      x1,y1,x2,y2=box[0:4]    
                      face_img=one_im[y1:y2,x1:x2,:]
                      emotion,scores=fer.predict_emotions(face_img,logits=True)
                      print(emotion,scores)

                  # Face detection
                  faces, is_null = extract_faces_mtcnn(ims, INPUT_SIZE)
                  print(f"Detected Faces")
                  if TIMING_VERBOSE:
                    time3 = time.time()
                    print('Time: ', time3 - time2) 

                  # Load the relevant network and get its predictions
                  model = get_emotion_predictor(MODEL_TYPE)
                  scores_real = hse_preds(faces, model, model_type=MODEL_TYPE)
                  scores_real[is_null == 1] = 0 # clear the predictions from frames w/o faces!
                  print("Got Network Predictions")
                  if TIMING_VERBOSE:
                    time4 = time.time()
                    print('Time: ', time4 - time3)

                  # Save outputs to a CSV
                  frames = np.arange(frame_now, frame_now + BATCH_NOW).reshape(BATCH_NOW, 1)
                  csv_save_HSE(labels=scores_real, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
                  print(f"Saved CSV to {save_path_now}!")

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

              # --- DEBUG ---
              faces = np.zeros([ims.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3], dtype=np.uint8)
              for enum_now, one_im in enumerate(ims):
                bounding_boxes=detect_face(one_im)
                bbox = bounding_boxes[0]
                box = bbox.astype(int)
                x1,y1,x2,y2=box[0:4]    
                face_img=one_im[y1:y2,x1:x2,:]
                face_img=cv2.resize(face_img, INPUT_SIZE)
                faces[enum_now, :, :, :] = face_img
              emotions,scores=fer.predict_multi_emotions(faces,logits=True)
              print(emotions,scores)

              # Face detection
              faces, is_null = extract_faces_mtcnn_new(ims, INPUT_SIZE)
              is_null = np.zeros(ims.shape[0])
              print(f"Detected Faces")
              if TIMING_VERBOSE:
                time3 = time.time()
                print('Time: ', time3 - time2) 

              # Load the relevant network and get its predictions
              model = get_emotion_predictor(MODEL_TYPE)
              scores_real = hse_preds_new(faces, model, model_type=MODEL_TYPE)
              scores_real[is_null == 1] = 0 # clear the predictions from frames w/o faces!
              print("Got Network Predictions")
              if TIMING_VERBOSE:
                time4 = time.time()
                print('Time: ', time4 - time3)

              # Save outputs to a CSV
              frames = np.arange(frame_now, frame_now + BATCH_NOW).reshape(BATCH_NOW, 1)
              csv_save_HSE(labels=scores_real, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
              print(f"Saved CSV to {save_path_now}!")

              frame_now = frame_now + BATCH_NOW
              frame_printing = frame_printing + BATCH_SIZE 

              # Reset ims to save space
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

 

