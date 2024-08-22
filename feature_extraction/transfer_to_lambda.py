# Check and load or generate dfs_openface
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_openface_{PAT_SHORT_NAME}.pkl'):
    # Generate dfs_openface if not already saved
    dfs_openface = get_dict_openface(OPENFACE_OUTPUT_DIRECTORY)
    dfs_openface = apply_function_to_dict(dfs_openface, only_successful_frames)
    save_var(dfs_openface, forced_name=f'dfs_openface_{PAT_SHORT_NAME}')
else:
    # Load dfs_openface if it already exists
    dfs_openface = load_var(f'dfs_openface_{PAT_SHORT_NAME}')

# Check and load or generate dfs_openface_extras
if not os.path.exists(RUNTIME_VAR_PATH + f'dfs_openface_extras_{PAT_NOW}.pkl'):
    # Generate dfs_openface_extras if not already saved
    dfs_openface_extras = get_dict_openface_extras(OPENFACE_OUTPUT_DIRECTORY)
    dfs_openface_extras = apply_function_to_dict(dfs_openface_extras, only_successful_frames)
    save_var(dfs_openface_extras, forced_name=f'dfs_openface_extras_{PAT_NOW}')
else:
    # Load dfs_openface_extras if it already exists
    dfs_openface_extras = load_var(f'dfs_openface_extras_{PAT_NOW}')


