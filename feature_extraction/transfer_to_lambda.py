import re

def print_invalid_time_rows(df):
    pattern = re.compile(r'^\d{2}:\d{2}:\d{2}$')
    invalid_rows = df[~df['Time Start'].astype(str).str.match(pattern)]
    print(invalid_rows)
