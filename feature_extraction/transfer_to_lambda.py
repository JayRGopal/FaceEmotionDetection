wget https://github.com/AcademySoftwareFoundation/openexr/archive/refs/tags/v2.5.7.tar.gz
tar -xzf v2.5.7.tar.gz
cd openexr-2.5.7

sudo apt update
sudo apt install -y cmake build-essential

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install

sudo ldconfig

