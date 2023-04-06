from utils import *


"""

Full Pipeline - OpenGraphAU

"""


# Set the parameters
START_FRAME = 300
NUM_TO_EXTRACT = 300
MODEL_TYPE = 'OpenGraphAU'
MODEL_BACKBONE = 'resnet50'
POST_PROCESSING_METHOD = 'KF'
VIDEO_PATH = 'Demo_Video.mp4'
SAVE_PATH = f'outputs/Demo_Output_{START_FRAME}_{NUM_TO_EXTRACT}.mp4'


# Extract video frames
(ims, im_test) = extract_images(path=VIDEO_PATH, start_frame=START_FRAME, num_to_extract=NUM_TO_EXTRACT)

# Detect a face in each frame
faces = detect_extract_faces(ims)

# Load the relevant network and get its predictions
net = load_network(model_type=MODEL_TYPE, backbone=MODEL_BACKBONE)
predictions = get_model_preds(faces, net, model_type=MODEL_TYPE)

# Post-processing
preds_post = postprocess_outs(predictions, method=POST_PROCESSING_METHOD)

# Create and download an output video
labels = extract_labels(ims, preds_post, model_type=MODEL_TYPE)
save_video_from_images(labels, video_name=SAVE_PATH, fps=30)


