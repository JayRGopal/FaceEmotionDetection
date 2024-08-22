# Check and load or generate openface_radius_dict
if not os.path.exists(RUNTIME_VAR_PATH + f'openface_radius_dict_{PAT_SHORT_NAME}.pkl'):
    # Generate openface_radius_dict if not already saved
    openface_radius_dict = time_splitter(openface_radius_dict, [5, 10])
    save_var(openface_radius_dict, forced_name=f'openface_radius_dict_{PAT_SHORT_NAME}')
else:
    # Load openface_radius_dict if it already exists
    openface_radius_dict = load_var(f'openface_radius_dict_{PAT_SHORT_NAME}')

# Check and load or generate hsemotion_radius_dict
if not os.path.exists(RUNTIME_VAR_PATH + f'hsemotion_radius_dict_{PAT_SHORT_NAME}.pkl'):
    # Generate hsemotion_radius_dict if not already saved
    hsemotion_radius_dict = time_splitter(hsemotion_radius_dict, [5, 10])
    save_var(hsemotion_radius_dict, forced_name=f'hsemotion_radius_dict_{PAT_SHORT_NAME}')
else:
    # Load hsemotion_radius_dict if it already exists
    hsemotion_radius_dict = load_var(f'hsemotion_radius_dict_{PAT_SHORT_NAME}')

# Check and load or generate opengraphau_radius_dict
if not os.path.exists(RUNTIME_VAR_PATH + f'opengraphau_radius_dict_{PAT_SHORT_NAME}.pkl'):
    # Generate opengraphau_radius_dict if not already saved
    opengraphau_radius_dict = time_splitter(opengraphau_radius_dict, [5, 10])
    save_var(opengraphau_radius_dict, forced_name=f'opengraphau_radius_dict_{PAT_SHORT_NAME}')
else:
    # Load opengraphau_radius_dict if it already exists
    opengraphau_radius_dict = load_var(f'opengraphau_radius_dict_{PAT_SHORT_NAME}')

# Check and load or generate openface_extras_radius_dict
if not os.path.exists(RUNTIME_VAR_PATH + f'openface_extras_radius_dict_{PAT_SHORT_NAME}.pkl'):
    # Generate openface_extras_radius_dict if not already saved
    openface_extras_radius_dict = time_splitter(openface_extras_radius_dict, [5, 10])
    save_var(openface_extras_radius_dict, forced_name=f'openface_extras_radius_dict_{PAT_SHORT_NAME}')
else:
    # Load openface_extras_radius_dict if it already exists
    openface_extras_radius_dict = load_var(f'openface_extras_radius_dict_{PAT_SHORT_NAME}')
