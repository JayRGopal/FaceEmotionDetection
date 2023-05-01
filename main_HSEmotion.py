from utils import *
from utilsHSE import *
import os
"""

Full Pipeline - HSEmotion

"""


# Set the parameters
START_FRAME = 0
BATCH_SIZE = 900
MODEL_TYPE = 'mobilenet_7.h5'
INPUT_SIZE = (224, 224)
VIDEO_DIRECTORY = os.path.abspath('inputs/')
FPS_EXTRACTING = 5 # we'll extract 5 fps


SAVE_PATH_FOLDER = lambda video_name: os.path.join(os.path.abspath('outputs_HSEmotion'), f'{video_name}')
SAVE_PATH = lambda save_path_folder, starter_frame: os.path.join(save_path_folder, f'{starter_frame}.csv')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# Loop through all videos
for i in all_videos:
  # Process the entirety of each video via a while loop!
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  frame_now = START_FRAME # this is what we save in outputs file
  frame_printing = START_FRAME # this is the "real" frame we are at
  im_test = tf.zeros(2) # placeholder
  fps = get_fps(path=video_path, extracting_fps=FPS_EXTRACTING)
  save_path_folder = SAVE_PATH_FOLDER(i)
  if not(os.path.exists(save_path_folder)):
    os.mkdir(save_path_folder)
  save_path_now = SAVE_PATH(save_path_folder, START_FRAME)

  while im_test.shape[0] != 0:
    # Extract video frames
    (ims, im_test) = extract_images(path=video_path, start_frame=frame_printing, num_to_extract=BATCH_SIZE, method='tensorflow', fps = FPS_EXTRACTING)
    BATCH_NOW = im_test.shape[0]
    if BATCH_NOW == 0:
      break
    print(f"Extracted Ims, Frames {frame_printing} to {frame_printing+BATCH_SIZE} in {i}")

    # Detect a face in each frame
    faces, is_null = extract_faces_mtcnn(ims, INPUT_SIZE)
    print(f"Detected Faces")

    # Load the relevant network and get its predictions
    model = get_emotion_predictor(MODEL_TYPE)
    scores_real = hse_preds(faces, model, model_type=MODEL_TYPE)
    scores_real[is_null == 1] = 0 # clear the predictions from frames w/o faces!
    print("Got Network Predictions")

    # TODO: Add some post-processing
    # print("Post Processing Complete")

    # Save outputs to a CSV
    frames = np.arange(frame_now, frame_now + BATCH_NOW).reshape(BATCH_NOW, 1)
    csv_save_HSE(labels=scores_real, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
    print(f"Saved CSV to {save_path_now}!")

    frame_now = frame_now + BATCH_NOW
    frame_printing = frame_printing + BATCH_SIZE

    # Skipping the annotated video for speed!
    # # Create and download an output video
    # labels = extract_labels(ims, preds_post, model_type=MODEL_TYPE)
    # save_video_from_images(labels, video_name=SAVE_PATH, fps=30)

  

