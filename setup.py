import os

def initial_setup():
    os.system("git clone https://github.com/JayRGopal/ME-GraphAU")
    os.system("pip install --user -r requirements.txt")
    os.rename('me-graphau', 'megraphau')
    return

# initial_setup()


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

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#down_now("megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth", '1wnJzvZ8bTR1yc4BhAiNaqU3HH10YX_cf')
#down_now("megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth", '178lhLCfPKOKlBLj2QgbqQDFEfoXMHEoD')
download_file_from_google_drive('1wnJzvZ8bTR1yc4BhAiNaqU3HH10YX_cf', 'megraphau/checkpoints/OpenGprahAU-ResNet50_first_stage.pth')
download_file_from_google_drive('178lhLCfPKOKlBLj2QgbqQDFEfoXMHEoD', 'megraphau/checkpoints/MEFARG_resnet50_BP4D_fold3.pth')
download_file("https://download.pytorch.org/models/resnet50-19c8e357.pth", "megraphau/checkpoints/resnet50-19c8e357.pth")



