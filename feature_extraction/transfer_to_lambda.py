# Step 1: Check if libIlmImf-2_5.so.25 is in the cvquant environment's lib directory
find $CONDA_PREFIX/lib -name "libIlmImf*.so*"

# Step 2: If the file exists, export the library path to make it accessible
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Step 3: Make the export command persistent by adding it to the Conda environment's activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Step 4: Reactivate the environment to apply the changes
conda deactivate
conda activate cvquant

# Step 5: Verify the library path and try running the program again
echo $LD_LIBRARY_PATH