export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

/home/jgopal/Desktop/OpenFace/build/bin/FeatureExtraction: error while loading shared libraries: libIlmImf-2_5.so.25: cannot open shared object file: No such file or directory
