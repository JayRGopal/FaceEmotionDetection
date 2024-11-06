# Step 1: Link libIlmImf.so.25 from the conda environment to /usr/local/lib
sudo ln -s $CONDA_PREFIX/lib/libIlmImf.so.25 /usr/local/lib/libIlmImf-2_5.so.25

# Step 2: Update the system's library cache
sudo ldconfig
