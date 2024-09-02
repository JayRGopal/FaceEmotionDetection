Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_GroupRPlot.py", line 350, in <module>
    plt.scatter(r_values_prefix_1, np.full_like(r_values_prefix_1, 1), c=[calculate_variance_percentage(load_var(f'predictions_{patient}_{PREFIX_1 if metric != "Pain" else PREFIX_1_PAIN}', RUNTIME_VAR_PATH)[f'{metric}']['y_true'], metric) for patient in patients], cmap='viridis', s=100, label=LABEL_1)
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/matplotlib/pyplot.py", line 3684, in scatter
    __ret = gca().scatter(
            ^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/matplotlib/__init__.py", line 1465, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 4670, in scatter
    self._parse_scatter_color_args(
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 4489, in _parse_scatter_color_args
    raise invalid_shape_exception(c.size, xsize) from err
ValueError: 'c' argument has 14 elements, which is inconsistent with 'x' and 'y' with size 11.
