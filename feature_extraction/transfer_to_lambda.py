import pandas as pd

# Function to correct the time format
def correct_time_format(time_str):
    # Convert to datetime to easily manipulate hour and minute
    time_val = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
    if time_val.hour == 0 and time_val.minute < 60:
        # This indicates the format was mm:ss, wrongly read as hh:mm
        corrected_time = pd.to_datetime(f"{time_val.minute}:{time_val.second}", format='%M:%S').time()
    else:
        # Correct time format
        corrected_time = time_val.time()
    return corrected_time

# Apply the correction to 'Time Start' and 'Time End'
df['Time Start'] = df['Time Start'].apply(correct_time_format)
df['Time End'] = df['Time End'].apply(correct_time_format)
