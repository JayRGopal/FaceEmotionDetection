# SAVE VARIABLES
import pickle


def get_var_name(our_variable):
    namespace = globals()
    for name, obj in namespace.items():
        if obj is our_variable:
            return name
    return None

# Save the dictionary to a file using pickle
def save_var(our_variable, RUNTIME_VAR_PATH=RUNTIME_VAR_PATH, forced_name=None):
  if forced_name is None:
    name_now = get_var_name(our_variable)
  else:
    name_now = forced_name

  with open(RUNTIME_VAR_PATH + f'{name_now}.pkl', 'wb') as file:
      pickle.dump(our_variable, file)

def load_var(variable_name, RUNTIME_VAR_PATH=RUNTIME_VAR_PATH):
  # Load from the file# LOAD THE OPENFACE DICTIONARY

dfs_openface = load_var(f'dfs_openface_{PAT_SHORT_NAME}')

  with open(RUNTIME_VAR_PATH + f'{variable_name}.pkl', 'rb') as file:
      return pickle.load(file)


NEXT CODE BLOCK:

dfs_openface = get_dict_openface(OPENFACE_OUTPUT_DIRECTORY)
dfs_openface = apply_function_to_dict(dfs_openface, only_successful_frames)

NEXT CODE BLOCK:

# SAVE THE OPENFACE DICTIONARY

save_var(dfs_openface, forced_name=f'dfs_openface_{PAT_SHORT_NAME}')


NEXT CODE BLOCK:

# LOAD THE OPENFACE DICTIONARY

dfs_openface = load_var(f'dfs_openface_{PAT_SHORT_NAME}')

NEXT CODE BLOCK:

dfs_openface_extras = get_dict_openface_extras(OPENFACE_OUTPUT_DIRECTORY)
dfs_openface_extras = apply_function_to_dict(dfs_openface_extras, only_successful_frames)

NEXT CODE BLOCK:

# SAVE THE OPENFACE EXTRAS DICTIONARY

save_var(dfs_openface_extras, forced_name=f'dfs_openface_extras_{PAT_NOW}')


NEXT CODE BLOCK:

# LOAD THE OPENFACE EXTRAS DICTIONARY

dfs_openface_extras = load_var(f'dfs_openface_extras_{PAT_NOW}')

