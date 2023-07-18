from utilsMMPose import *


FOLDER_WITH_FOLDERS_WITH_CSVs = os.path.abspath('merger/')
PATH_TO_PICKLE = os.path.abspath('MMPose_info/merged_csvs.pkl') 

concatenate_csvs_in_folders(FOLDER_WITH_FOLDERS_WITH_CSVs, PATH_TO_PICKLE)
