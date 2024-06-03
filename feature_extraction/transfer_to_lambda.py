def fill_empty_dfs_lists(dictionary):
  # when we do emotion processing, some dfs will have ZERO successful frames,
  # leading to ZERO events, and an empty df.
  # we need to fill the empty dfs with a df with all 0s

  non_empty_dfs = [[df for df in df_list if not df.empty] for df_list in dictionary.values()]

  if not non_empty_dfs:
      return dictionary  # Return the original dictionary if all DataFrames are empty

  non_empty_df = non_empty_dfs[0][0]  # Choose the first non-empty DataFrame as replacement

  modified_dictionary = {}
  for key, df_list in dictionary.items():
      modified_df_list = []
      for df in df_list:
        if df.empty:
            modified_df = pd.DataFrame(0, index=non_empty_df.index, columns=non_empty_df.columns)
            # Preserve string columns from non-empty DataFrame
            for column in non_empty_df.columns:
                if non_empty_df[column].dtype == object:
                    modified_df[column] = non_empty_df[column]
        else:
            modified_df = df.copy()

        modified_df_list.append(modified_df)

      modified_dictionary[key] = modified_df_list

  return modified_dictionary