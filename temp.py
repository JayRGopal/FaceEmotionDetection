Analyzing patient S23_203 with method OpenFace
Traceback (most recent call last):
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_NoBootstraps_Comprehensive.py", line 1229, in <module>
    main()
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_NoBootstraps_Comprehensive.py", line 1204, in main
    analyze_single_patient(patient_id, patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_binary=True)
  File "/home/jgopal/Desktop/FaceEmotionDetection/main_NoBootstraps_Comprehensive.py", line 281, in analyze_single_patient
    model.fit(X_train, y_train)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/base.py", line 1473, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py", line 1991, in fit
    fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer)(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/utils/parallel.py", line 74, in __call__
    return super().__call__(iterable_with_config)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/joblib/parallel.py", line 1918, in __call__
    return output if self.return_generator else list(output)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/joblib/parallel.py", line 1847, in _get_sequential_output
    res = func(*args, **kwargs)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/utils/parallel.py", line 136, in __call__
    return self.function(*args, **kwargs)
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py", line 748, in _log_reg_scoring_path
    coefs, Cs, n_iter = _logistic_regression_path(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py", line 507, in _logistic_regression_path
    ) = _fit_liblinear(
  File "/home/jgopal/miniconda3/envs/cvquant/lib/python3.10/site-packages/sklearn/svm/_base.py", line 1173, in _fit_liblinear
    raise ValueError(
ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: -1.0
