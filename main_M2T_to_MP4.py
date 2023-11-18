import os
from pathlib import Path
import subprocess
import moviepy.editor as moviepy
import shutil

# Define patient directory
ptDir = Path('J:\\S23_203\\NKT')
target = 'E:\\S203_MP4'


# Convert M2T files to MP4
def convert_to_mp4(m2t_file):
    name, ext = os.path.splitext(m2t_file)
    out_name = name + ".mp4"
    cmd = f'ffmpeg -i {m2t_file} -c:v libx264 -preset veryfast -crf 18 -c:a copy {out_name}'
    subprocess.run(cmd, shell=True)
    print("Finished converting {}".format(m2t_file))

start_dir = ptDir
for path, folder, files in os.walk(start_dir):
    for file in files:
        if file.endswith('.m2t'):
            name, ext = os.path.splitext(file)
            check_name = name + ".mp4"
            if os.path.isfile(os.path.join(path, check_name)):
                print("Already processed: %s" % file)
            else:
                print("Found file to process: %s" % file)
                convert_to_mp4(os.path.join(path, file))
        else:
            pass


# Copy MP4 files to another destination

source = ptDir
start_dir = source
for path, folder, files in os.walk(start_dir):
    for file in files:
        if file.endswith('.mp4'):
            if os.path.isfile(os.path.join(target,file)):
                print("Already copied: %s" % file)
            else:
                shutil.copy(os.path.join(path, file), os.path.join(target,file))
                print("Copying and deleting: %s" % file)
                os.remove(os.path.join(path, file))
        else:
            pass