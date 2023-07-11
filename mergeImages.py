from utilsMMPose import *


FOLDER_WITH_FOLDERS_WITH_IMAGES = os.path.abspath('merger/')
PATH_TO_NEW_VIDEO = os.path.abspath('inputs/merged_video.mp4') 

merge_images_to_video(FOLDER_WITH_FOLDERS_WITH_IMAGES, PATH_TO_NEW_VIDEO, os.path.abspath('MMPose_info/merged_image_order.json'))

