# ALPHA PARAMETER SEARCH FOR LASSO - RUN THIS FIRST!

all_metrics = [col for col in df_moodTracking.columns if col != 'Datetime']

FILE_ENDING = '.png'

# We are just searching using lasso regression
#RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OFAUHSE_L_', 'OGAUHSE_L_', 'HSE_L_', 'ALL_L_']
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
#RESULTS_PREFIX_LIST = ['OGAUHSE_L_']


EMOTIONS_FOR_SEARCH = ['Mood'] # We are just searching on Mood
TIME_WINDOW_FOR_SEARCH = '180' # We are just searching 3 hours

# List of alpha values to search through
#ALPHAS_FOR_SEARCH = np.arange(0, 1.6, 0.1)
ALPHAS_FOR_SEARCH = np.arange(0, 5, 0.2)
#ALPHAS_FOR_SEARCH = np.arange(0, 10, 0.2)

# This will populate with the best alphas for each prefix in RESULTS_PREFIX_LIST
best_alphas_lasso = {}

for RESULTS_PREFIX in RESULTS_PREFIX_LIST:
  do_lasso = False
  do_ridge = False

  if '_L_' in RESULTS_PREFIX:
    do_lasso = True

  if '_R_' in RESULTS_PREFIX:
    do_ridge = True


  if 'OF_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+f'experimental3_openface_0.5_hours.xlsx'
    vectors_now = openface_vectors_dict
    method_now = 'OpenFace'

  elif 'OGAU_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+'opengraphau_0.5_hours.xlsx'
    vectors_now = opengraphau_vectors_dict
    method_now = 'OpenGraphAU'

  elif 'OFAUHSE_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+'ofauhse_0.5_hours.xlsx'
    vectors_now = ofauhsemotion_vectors_dict
    method_now = 'OFAU+HSE'

  elif 'OGAUHSE_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+'ogauhse_0.5_hours.xlsx'
    vectors_now = ogauhsemotion_vectors_dict
    method_now = 'OGAU+HSE'

  elif 'HSE_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+'hsemotion_0.5_hours.xlsx'
    vectors_now = hsemotion_vectors_dict
    method_now = 'HSEmotion'

  elif 'ALL_' in RESULTS_PREFIX:
    spreadsheet_path = FEATURE_LABEL_PATH+'all_0.5_hours.xlsx'
    vectors_now = all_vectors_dict
    method_now = 'ALL(OF+OG+HSE)'


  # Let's put each setting in its own folder!
  os.makedirs(RESULTS_PATH_BASE + 'SEARCH_Alpha_Lasso/' + RESULTS_PREFIX, exist_ok=True)
  results_prefix_unmodified = RESULTS_PREFIX
  RESULTS_PREFIX = 'SEARCH_Alpha_Lasso/' + RESULTS_PREFIX + '/' + RESULTS_PREFIX

  # This will store the best R, averaged across all metrics we're testing, for each alpha
  pearson_r_list = []

  for alpha_now in ALPHAS_FOR_SEARCH:

    avg_best_R = 0

    # Loop through EMOTIONS_FOR_SEARCH
    for metric in EMOTIONS_FOR_SEARCH:
      print('METRIC NOW: ', metric)
      vectors_return, y = extractOneMetric(metric, vectors_now=vectors_now)
      
      # Limit to just one time window for alpha search
      tmp_vectors = vectors_return
      vectors_return = {}
      vectors_return[TIME_WINDOW_FOR_SEARCH] = tmp_vectors[TIME_WINDOW_FOR_SEARCH]
      del tmp_vectors

      scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
      scores_r, preds_r, _, models_r = linRegOneMetric(vectors_return, y, randShuffle=True, alpha=alpha_now)

      # make scatterplots
      randShuffleR, _, _ = plot_scatterplots(preds_r, y, f'{metric} Random Shuffle', RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterRand_{alpha_now}{FILE_ENDING}')
      r_list, p_list, scatterFig = plot_scatterplots(preds, y, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterplots_{alpha_now}{FILE_ENDING}', randShuffleR=randShuffleR)

      # Determine our best time radius for this metric based on Pearson's R
      best_time_radius = list(scores.keys())[np.argmax(r_list)]
      best_mse_list = scores[best_time_radius]
      best_avg_mse = np.mean(scores[best_time_radius])
      best_pearson_r = r_list[np.argmax(r_list)]

      # Add to our avg best R
      avg_best_R = avg_best_R + best_pearson_r

      # bar plot for pearson r
      rPlotFig = make_r_barplot(r_list, list(scores.keys()), metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_R_{alpha_now}{FILE_ENDING}', method_now=method_now)

      # make MSE plot
      MSEPlotFig = make_mse_boxplot(scores, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_MSE_{alpha_now}{FILE_ENDING}', method_now=method_now)

    # Add one R value for this alpha value to pearson_r_list
    avg_best_R = avg_best_R / len(EMOTIONS_FOR_SEARCH)
    pearson_r_list.append(avg_best_R)

  # Plot R vs. alpha for this setting
  plot_pearsons_r_vs_alpha(pearson_r_list=pearson_r_list, ALPHAS_FOR_SEARCH=ALPHAS_FOR_SEARCH, method=method_now, save_path=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_Alpha_Search{FILE_ENDING}')

  # Find best alpha for this setting
  best_index_of_alpha = np.argmax(pearson_r_list)
  best_alpha_value = ALPHAS_FOR_SEARCH[best_index_of_alpha]
  best_alphas_lasso[results_prefix_unmodified] = best_alpha_value
