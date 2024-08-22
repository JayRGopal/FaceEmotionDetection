Do the same for this


dfs_hsemotion = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_hse.csv')
dfs_hsemotion = apply_function_to_dict(dfs_hsemotion, only_successful_frames)


OPENGRAPHAU_THRESHOLD = 0.5
dfs_opengraphau = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_ogau.csv')
dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, create_binary_columns, threshold=OPENGRAPHAU_THRESHOLD)
dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, only_successful_frames)
dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, remove_columns_ending_with_r)


# SAVE THE HSEMOTION AND OPENGRAPHAU DICTIONARIES

save_var(dfs_hsemotion, forced_name=f'dfs_hsemotion_{PAT_SHORT_NAME}')

save_var(dfs_opengraphau, forced_name=f'dfs_opengraphau_{PAT_SHORT_NAME}')


# LOAD THE HSEMOTION AND OPENGRAPHAU DICTIONARIES

dfs_hsemotion = load_var(f'dfs_hsemotion_{PAT_SHORT_NAME}')

dfs_opengraphau = load_var(f'dfs_opengraphau_{PAT_SHORT_NAME}')
