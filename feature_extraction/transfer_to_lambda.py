sudo apt install -y build-essential yasm pkg-config libx264-dev libx265-dev libfdk-aac-dev
wget https://ffmpeg.org/releases/ffmpeg-4.3.1.tar.bz2
tar xjf ffmpeg-4.3.1.tar.bz2
cd ffmpeg-4.3.1
./configure --enable-gpl --enable-libx264 --enable-libx265 --enable-nonfree
make
sudo make install