# Check and load or generate dfs_hsemotion
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_hsemotion_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_hsemotion if not already saved
    dfs_hsemotion = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_hse.csv')
    dfs_hsemotion = apply_function_to_dict(dfs_hsemotion, only_successful_frames)
    save_var(dfs_hsemotion, forced_name=f'dfs_hsemotion_{PAT_SHORT_NAME}')
else:
    # Load dfs_hsemotion if it already exists
    dfs_hsemotion = load_var(f'dfs_hsemotion_{PAT_SHORT_NAME}')

# Check and load or generate dfs_opengraphau
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_opengraphau_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_opengraphau if not already saved
    OPENGRAPHAU_THRESHOLD = 0.5
    dfs_opengraphau = get_dict(COMBINED_OUTPUT_DIRECTORY, file_now='outputs_ogau.csv')
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, create_binary_columns, threshold=OPENGRAPHAU_THRESHOLD)
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, only_successful_frames)
    dfs_opengraphau = apply_function_to_dict(dfs_opengraphau, remove_columns_ending_with_r)
    save_var(dfs_opengraphau, forced_name=f'dfs_opengraphau_{PAT_SHORT_NAME}')
else:
    # Load dfs_opengraphau if it already exists
    dfs_opengraphau = load_var(f'dfs_opengraphau_{PAT_SHORT_NAME}')