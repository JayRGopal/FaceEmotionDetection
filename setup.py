import os

os.system("git clone https://github.com/cvi-szu/me-graphau")
os.system("pip install --user -r requirements.txt")
os.rename('me-graphau', 'megraphau')


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

down_now("me-graphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth", '1wnJzvZ8bTR1yc4BhAiNaqU3HH10YX_cf')
down_now("me-graphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth", '178lhLCfPKOKlBLj2QgbqQDFEfoXMHEoD')
download_file("https://download.pytorch.org/models/resnet50-19c8e357.pth", "me-graphau/checkpoints/resnet50-19c8e357.pth")



