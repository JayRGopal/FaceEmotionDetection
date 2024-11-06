# Step 1: Clear LD_LIBRARY_PATH to start fresh
unset LD_LIBRARY_PATH

# Step 2: Set it only once with the current Conda environment's lib directory
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# Step 3: Confirm the path now contains only one entry
echo $LD_LIBRARY_PATH