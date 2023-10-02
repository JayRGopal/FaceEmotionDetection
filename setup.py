import os

def initial_setup():

    # requirements
    os.system("pip install --user -r requirements.txt")
    os.system("pip3 install --user -r requirements.txt")
    
    # ME-GraphAU Install
    if not(os.path.exists('megraphau')):
        os.system("git clone https://github.com/JayRGopal/ME-GraphAU")
        os.rename('ME-GraphAU', 'megraphau')
    
    # Yolo_Tracking Install
    if not(os.path.exists('yolo_tracking')):
        os.system("git clone https://github.com/JayRGopal/yolo_tracking")
        os.chdir('yolo_tracking')
        os.system("pip install -e .")
        os.chdir('..')


    # NEMO Install
    # if not os.path.exists("scripts_nemo_asr/transcribe_speech.py"):
    #     os.system("wget -P scripts_nemo_asr/ https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/transcribe_speech.py")

    # if not os.path.exists("scripts_nemo_asr/speech_to_text_eval.py"):
    #     os.system("wget -P scripts_nemo_asr/ https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/speech_to_text_eval.py")
    
    # os.system('pip3 install Cython')
 
    # os.system('python3 -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.18.0#egg=nemo_toolkit')

    # Note: You may also need visualstudio.microsoft.com/visual-cpp-build-tools/
    # Note: We also had to clone the nemo repo and pip install requirements on the CPU workstation

    # MMPose Install
    os.system('pip install numpy==1.23.5')
    os.system("pip install -U openmim")
    os.system("pip3 install mmpose")
    os.system("python3 -m mim install mmengine")
    os.system('python3 -m mim install "mmcv>=2.0.0"')
    os.system('python3 -m mim install "mmdet>=3.0.0"')
    os.system('python3 -m mim install "mmcls>=1.0.0rc5"')
    os.system('python3 -m mim install "mmpretrain>=1.0.0"') # for ViT Pose
    if not(os.path.exists('mmpose')):
        os.system("git clone https://github.com/JayRGopal/mmpose")
    os.system('cd mmpose')
    os.system('pip install -r requirements.txt')
    os.system('pip install -v -e .')
    os.system('cd ..')
    os.system('pip install numpy==1.23.5')

    return

def gpu_setup():
    # May need to adjust this based on workstation
    os.system("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    return

if __name__ == '__main__':
    initial_setup()

    # Uncomment for GPU
    #gpu_setup()


import requests

def down_now(save_path, file_id):
    # Construct the direct download link
    download_link = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Call the requests library to download the file
    response = requests.get(download_link)
    if response.status_code == 200:
        # Save the downloaded file to the specified path
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded and saved to {save_path}.")
    else:
        print("An error occurred while downloading the file.")
    
    return

def download_file(link: str, output_path: str) -> None:
    
    # Download the file using requests with streaming and chunked writing
    with requests.get(link, stream=True) as response:
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File downloaded and saved to {output_path}")
        else:
            print("An error occurred while downloading the file.")

import gdown

def gdownload(fid, dest):
    url = f'https://drive.google.com/uc?id={fid}'
    destination = dest
    gdown.download(url, destination, quiet=False)
    return



if __name__ == '__main__':
    if not(os.path.exists("megraphau/checkpoints/swin_base_patch4_window7_224.pth")): 
        gdownload('1U_PEhO5ymJc77kNtVrMja_cRcC0UpfS4', "megraphau/checkpoints/swin_base_patch4_window7_224.pth")

    if not(os.path.exists("megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth")): 
        gdownload('1_sJpvGLbZxoFenQFT7qYtTw91tEKSe0u', "megraphau/checkpoints/OpenGprahAU-SwinB_first_stage.pth")
    
    if not(os.path.exists('megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')):
        gdownload('1zuIZXz7MPsKOlDJMjju8oMYZRifl9sHt', 'megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')
        #gdownload('1wnJzvZ8bTR1yc4BhAiNaqU3HH10YX_cf', 'megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')

    # if not(os.path.exists('megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth')): 
    #     gdownload('178lhLCfPKOKlBLj2QgbqQDFEfoXMHEoD', 'megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth')

    if not(os.path.exists("megraphau/checkpoints/resnet50-19c8e357.pth")): 
        gdownload('1i68ZxnsI7Hw6vxcFy_IXXfQ05supUDhp', "megraphau/checkpoints/resnet50-19c8e357.pth")
        #download_file("https://download.pytorch.org/models/resnet50-19c8e357.pth", "megraphau/checkpoints/resnet50-19c8e357.pth")


    if not(os.path.exists("MMPose_models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")): 
        gdownload('1xCcdCFslOp2K6GNg9ep6RdYyOeue2V7g', "MMPose_models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
        #download_file('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', "MMPose_models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")


