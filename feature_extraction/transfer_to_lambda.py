And again for this (no need to give me those loading and saving functions you're not changing!')

openface_radius_dict = time_splitter(openface_radius_dict, [5, 10])
hsemotion_radius_dict = time_splitter(hsemotion_radius_dict, [5, 10])
opengraphau_radius_dict = time_splitter(opengraphau_radius_dict, [5, 10])
openface_extras_radius_dict = time_splitter(openface_extras_radius_dict, [5, 10])


# SAVE VARIABLES - EMOTION DETECTION & AFFECT

save_var(openface_radius_dict, forced_name=f'openface_radius_dict_{PAT_SHORT_NAME}')

save_var(hsemotion_radius_dict, forced_name=f'hsemotion_radius_dict_{PAT_SHORT_NAME}')

save_var(opengraphau_radius_dict, forced_name=f'opengraphau_radius_dict_{PAT_SHORT_NAME}')

save_var(openface_extras_radius_dict, forced_name=f'openface_extras_radius_dict_{PAT_SHORT_NAME}')



# LOAD VARIABLES - EMOTION DETECTION & AFFECT

openface_radius_dict = load_var(f'openface_radius_dict_{PAT_SHORT_NAME}')

hsemotion_radius_dict = load_var(f'hsemotion_radius_dict_{PAT_SHORT_NAME}')

opengraphau_radius_dict = load_var(f'opengraphau_radius_dict_{PAT_SHORT_NAME}')

openface_extras_radius_dict = load_var(f'openface_extras_radius_dict_{PAT_SHORT_NAME}')
