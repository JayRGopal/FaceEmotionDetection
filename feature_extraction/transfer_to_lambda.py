Danny_Labels.loc[:, Danny_Labels.columns.str.contains('Time')] = Danny_Labels.loc[:, Danny_Labels.columns.str.contains('Time')].applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)
