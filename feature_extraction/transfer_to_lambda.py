Collecting numpy>=1.9 (from py-feat==0.6.2)
  Using cached numpy-1.23.5.tar.gz (10.7 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [33 lines of output]
      Traceback (most recent call last):
        File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 353, in <module>
          main()
        File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 112, in get_requires_for_build_wheel
          backend = _build_backend()
                    ^^^^^^^^^^^^^^^^
        File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
          obj = import_module(mod_path)
                ^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/jgopal/miniconda3/envs/pyfeat/lib/python3.12/importlib/__init__.py", line 90, in import_module
          return _bootstrap._gcd_import(name[level:], package, level)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
        File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
        File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
        File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
        File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
        File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
        File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
        File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
        File "<frozen importlib._bootstrap_external>", line 995, in exec_module
        File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
        File "/tmp/pip-build-env-ih8jkvv7/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 16, in <module>
          import setuptools.version
        File "/tmp/pip-build-env-ih8jkvv7/overlay/lib/python3.12/site-packages/setuptools/version.py", line 1, in <module>
          import pkg_resources
        File "/tmp/pip-build-env-ih8jkvv7/overlay/lib/python3.12/site-packages/pkg_resources/__init__.py", line 2172, in <module>
          register_finder(pkgutil.ImpImporter, find_on_path)
                          ^^^^^^^^^^^^^^^^^^^
      AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
