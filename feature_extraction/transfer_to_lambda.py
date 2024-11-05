wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar xzf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure
make
sudo make install


./configure --enable-gpl --enable-libx264 --enable-libx265 --enable-nonfree --disable-asm
make