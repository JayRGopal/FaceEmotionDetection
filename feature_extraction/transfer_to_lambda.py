Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_PyFeat.py", line 1, in <module>
    from feat import Detector
  File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.11/site-packages/feat/__init__.py", line 11, in <module>
    from .data import Fex
  File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.11/site-packages/feat/data.py", line 30, in <module>
    from feat.utils.stats import wavelet, calc_hist_auc, cluster_identities
  File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.11/site-packages/feat/utils/stats.py", line 7, in <module>
    from scipy.integrate import simps
ImportError: cannot import name 'simps' from 'scipy.integrate' (/home/jgopal/miniconda3/envs/pyfeat/lib/python3.11/site-packages/scipy/integrate/__init__.py)
