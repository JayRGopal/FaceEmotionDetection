/tmp/ipykernel_20173/1959524370.py:1: DeprecationWarning: 
Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
(to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
but was not found to be installed on your system.
If this would cause problems for you,
please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
        
  import pandas as pd
/tmp/ipykernel_20173/1959524370.py:37: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  column_df[column_df.columns[0]] = pd.to_datetime(column_df[column_df.columns[0]], errors='coerce')
/tmp/ipykernel_20173/1959524370.py:37: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  column_df[column_df.columns[0]] = pd.to_datetime(column_df[column_df.columns[0]], errors='coerce')
/tmp/ipykernel_20173/1959524370.py:37: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  column_df[column_df.columns[0]] = pd.to_datetime(column_df[column_df.columns[0]], errors='coerce')