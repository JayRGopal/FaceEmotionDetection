import os

def initial_setup():

    # requirements
    #os.system("pip install --user -r requirements.txt")

    # ME-GraphAU Install
    if not(os.path.exists('megraphau')):
        os.system("git clone https://github.com/JayRGopal/ME-GraphAU")
        os.rename('me-graphau', 'megraphau')

    # NEMO Install
    if not os.path.exists("scripts/transcribe_speech.py"):
        os.system("wget -P scripts_nemo_asr/ https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/transcribe_speech.py")

    if not os.path.exists("scripts/speech_to_text_eval.py"):
        os.system("wget -P scripts_nemo_asr/ https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/speech_to_text_eval.py")

    return


initial_setup()


def gpu_setup():
    # May need to adjust this based on workstation
    os.system("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
    return

# Uncomment for GPU
# gpu_setup()

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

def download_file(link: str, output_dir: str) -> None:
    # Extract the file name from the link
    file_name = link.split("/")[-1]

    # Download the file using requests
    response = requests.get(link)
    if response.status_code == 200:
        # Save the downloaded file to the output directory
        with open(output_dir, "wb") as f:
            f.write(response.content)
        print(f"File downloaded and saved to {output_dir}")
    else:
        print("An error occurred while downloading the file.")
    return

import gdown

def gdownload(fid, dest):
    url = f'https://drive.google.com/uc?id={fid}'
    destination = dest
    gdown.download(url, destination, quiet=False)
    return

if not(os.path.exists('megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')):
    gdownload('1wnJzvZ8bTR1yc4BhAiNaqU3HH10YX_cf', 'megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')

if not(os.path.exists('megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth')): 
    gdownload('178lhLCfPKOKlBLj2QgbqQDFEfoXMHEoD', 'megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth')

if not(os.path.exists("megraphau/checkpoints/resnet50-19c8e357.pth")): 
    download_file("https://download.pytorch.org/models/resnet50-19c8e357.pth", "megraphau/checkpoints/resnet50-19c8e357.pth")



