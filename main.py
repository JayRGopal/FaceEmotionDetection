from utils import *
import os
"""

Full Pipeline - OpenGraphAU

"""


# Set the parameters
START_FRAME = 200
BATCH_SIZE = 2
MODEL_TYPE = 'OpenGraphAU'
MODEL_BACKBONE = 'resnet50'
POST_PROCESSING_METHOD = 'EMA'
VIDEO_DIRECTORY = os.path.abspath('inputs/')

SAVE_PATH_FOLDER = lambda video_name: os.path.join(os.path.abspath('outputs'), f'{video_name}')
SAVE_PATH = lambda save_path_folder, starter_frame: os.path.join(save_path_folder, f'{starter_frame}.csv')

# Get the list of all videos in the given directory
all_videos = [vid for vid in os.listdir(VIDEO_DIRECTORY) if vid[0:1] != '.']

# Loop through all videos
for i in all_videos:
  # Process the entirety of each video via a while loop!
  video_path = os.path.join(VIDEO_DIRECTORY, i)
  frame_now = START_FRAME
  im_test = torch.zeros(2) # placeholder
  fps = get_fps(path=video_path)
  save_path_folder = SAVE_PATH_FOLDER(i)
  if not(os.path.exists(save_path_folder)):
    os.mkdir(save_path_folder)
  save_path_now = SAVE_PATH(save_path_folder, START_FRAME)

  while im_test.size()[0] != 0:
    # Extract video frames
    (ims, im_test) = extract_images(path=video_path, start_frame=frame_now, num_to_extract=BATCH_SIZE)
    print(f"Extracted Ims, Frames {frame_now} to {frame_now+BATCH_SIZE} in {i}")

    # Detect a face in each frame
    faces, is_null = detect_extract_faces(ims)
    print(f"Detected Faces")

    # Load the relevant network and get its predictions
    net = load_network(model_type=MODEL_TYPE, backbone=MODEL_BACKBONE)
    predictions = get_model_preds(faces, net, model_type=MODEL_TYPE)
    predictions[is_null == 1] = 0 # clear the predictions from frames w/o faces!
    print("Got Network Predictions")

    # Post-processing
    preds_post = postprocess_outs(predictions, method=POST_PROCESSING_METHOD)
    # TODO: How do we deal will null frames in post-processing?
    print("Post Processing Complete")

    # Save outputs to a CSV
    frames = np.arange(frame_now, frame_now + BATCH_SIZE).reshape(BATCH_SIZE, 1)
    csv_save(labels=preds_post, is_null=is_null, frames=frames, save_path=save_path_now, fps=fps)
    print(f"Saved CSV to {save_path_now}!")

    frame_now = frame_now + BATCH_SIZE

    # Skipping the annotated video for speed!
    # # Create and download an output video
    # labels = extract_labels(ims, preds_post, model_type=MODEL_TYPE)
    # save_video_from_images(labels, video_name=SAVE_PATH, fps=30)

  

