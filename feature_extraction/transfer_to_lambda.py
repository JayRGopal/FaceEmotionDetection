Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_FeatureExtractionInpatient.py", line 3069, in <module>
    scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_FeatureExtractionInpatient.py", line 2117, in linRegOneMetric
    preds[i] = cross_val_predict(model, vectors_dict[i], y_using, cv=vectors_dict[i].shape[0])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 1284, in cross_val_predict
    predictions = parallel(
                  ^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 67, in __call__
    return super().__call__(iterable_with_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/joblib/parallel.py", line 1863, in __call__
    return output if self.return_generator else list(output)
                                                ^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/joblib/parallel.py", line 1792, in _get_sequential_output
    res = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 129, in __call__
    return self.function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 1369, in _fit_and_predict
    estimator.fit(X_train, y_train, **fit_params)
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/base.py", line 1351, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/linear_model/_coordinate_descent.py", line 955, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1192, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1003, in check_array
    _assert_all_finite(
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 126, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "/home/jgopal/miniconda3/envs/featureExtract/lib/python3.11/site-packages/sklearn/utils/validation.py", line 175, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
Lasso does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
