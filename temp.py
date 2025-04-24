Found 9 patient folders: ['S23_203', 'S23_174', 'S24_227', 'S24_218', 'S24_217', 'S24_219', 'S24_226', 'S23_207', 'S23_199']

Processing internal state: Mood
  Processing feature set: OpenFace
    Processing time window: 15 minutes
      Found data for 9 patients
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_NoBootstraps_AcrossPatients.py", line 138, in <module>
    model.fit(X_train, y_train)
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/base.py", line 1351, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py", line 1587, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/base.py", line 645, in _validate_data
    X = check_array(X, input_name="X", **check_X_params)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1003, in check_array
    _assert_all_finite(
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 126, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 175, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
LassoCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
