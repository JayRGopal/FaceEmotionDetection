import re
Danny_Labels.loc[:, Danny_Labels.columns.str.contains('Time')] = Danny_Labels.loc[:, Danny_Labels.columns.str.contains('Time')].applymap(lambda x: re.sub(r'(?<=:)(\d)(?=:)', r'0\1', x) if isinstance(x, str) else x)
