# GENERATE ALL PLOTS! ONE CODE BLOCK

if 'best_alphas_lasso' not in globals():
    raise NameError("GO RUN THE LASSO ALPHA PARAMETER SEARCH BLOCK FIRST!")

# if 'best_alphas_ridge' not in globals():
#     raise NameError("GO RUN THE RIDGE ALPHA PARAMETER SEARCH BLOCK FIRST!")


#all_metrics = [col for col in df_moodTracking.columns if col != 'Datetime']
#all_metrics = ['Mood', 'Anxiety', 'Hunger']
all_metrics = ['Mood']


FILE_ENDING = '.png'
# RESULTS_PREFIX_LIST = ['OF_', 'OGAU_', 'OFAUHSE_', 'OGAUHSE_', 'HSE_', 'ALL_',
#                        'OF_L_', 'OGAU_L_', 'OFAUHSE_L_', 'OGAUHSE_L_', 'HSE_L_', 'ALL_L_',
#                        'OF_R_', 'OGAU_R_', 'OFAUHSE_R_', 'OGAUHSE_R_', 'HSE_R_', 'ALL_R_']

# RESULTS_PREFIX_LIST = ['OF_L_', 'OGAUHSE_L_', 'OGAU_L_', 'OFAUHSE_L_', 'HSE_L_', 'ALL_L_']

RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']


# Do we remove ground truth labels that are over 2 standard deviations from the mean?
REMOVE_OUTLIERS = False


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
  os.makedirs(RESULTS_PATH_BASE + RESULTS_PREFIX, exist_ok=True)
  results_prefix_unmodified = RESULTS_PREFIX
  RESULTS_PREFIX = RESULTS_PREFIX + '/' + RESULTS_PREFIX


  # Loop through metrics (Anxiety, Depression, Mood, etc.)
  for metric in all_metrics:
    print('METRIC NOW: ', metric)
    if do_lasso:
      alpha_now = best_alphas_lasso[results_prefix_unmodified]
    elif do_ridge:
      alpha_now = best_alphas_ridge[results_prefix_unmodified]
    else:
      # Neither lasso nor ridge, so alpha is irrelevant
      alpha_now = 1.0

    vectors_return, y = extractOneMetric(metric, vectors_now=vectors_now, remove_outliers=REMOVE_OUTLIERS)
    scores, preds, y, models = linRegOneMetric(vectors_return, y, do_lasso=do_lasso, do_ridge=do_ridge, alpha=alpha_now)
    scores_r, preds_r, _, models_r = linRegOneMetric(vectors_return, y, randShuffle=True, alpha=alpha_now)

    # make scatterplots
    randShuffleR, _, _ = plot_scatterplots(preds_r, y, f'{metric} Random Shuffle', RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterRand{FILE_ENDING}')
    r_list, p_list, scatterFig = plot_scatterplots(preds, y, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_scatterplots{FILE_ENDING}', randShuffleR=randShuffleR)

    # Determine our best time radius for this metric based on Pearson's R
    best_time_radius = list(scores.keys())[np.argmax(r_list)]
    best_mse_list = scores[best_time_radius]
    best_avg_mse = np.mean(scores[best_time_radius])
    best_pearson_r = r_list[np.argmax(r_list)]

    # bar plot for pearson r
    rPlotFig = make_r_barplot(r_list, list(scores.keys()), metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_R{FILE_ENDING}', method_now=method_now)

    # make MSE plot
    MSEPlotFig = make_mse_boxplot(scores, metric, RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_MSE{FILE_ENDING}', method_now=method_now)

    # Feature ablation
    #feat_ab_scores, feat_ab_prs = featureAblate(vectors_return[best_time_radius], y, do_lasso=do_lasso, do_ridge=do_ridge)

    #top_indices, featAbMSEFig = plotFeatAbMSEs(feat_ab_scores, best_mse_list, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_MSEs{FILE_ENDING}', spreadsheet_path=spreadsheet_path)
    #plotFeatAbPRs(feat_ab_prs, best_pearson_r, metric, best_time_radius, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_featAblate_R{FILE_ENDING}', spreadsheet_path=spreadsheet_path)

    # extract just ONE scatterplot (the best pearson's R) and save it individually
    plt.rcParams['lines.markersize'] = 9
    _, _, bestScatterFig = plot_predictions(y, preds[best_time_radius], randShuffleR=randShuffleR[np.argmax(r_list)], ax=None, time_rad=best_time_radius, metric=metric)
    bestScatterFig.savefig(RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_linReg_bestScatter{FILE_ENDING}', bbox_inches='tight')

    # Plot top n features vs. self-reported scores
    #PLOT_NOW = 3
    #plot_feat_scatterplots(vectors_array=vectors_return[best_time_radius], y=y, feat_ind_list=top_indices[:PLOT_NOW], metric=metric, savepath=RESULTS_PATH_BASE + f'{RESULTS_PREFIX}{metric}_topFeats{FILE_ENDING}', spreadsheet_path=spreadsheet_path)
