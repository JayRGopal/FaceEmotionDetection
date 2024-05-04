def correct_time_format(time_str):
    
    # Convert to datetime to easily manipulate hour and minute
    time_str = str(time_str).replace(' ', '')
    time_val = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
    if time_val.hour == 0 and time_val.minute < 60:
        # This indicates the format was mm:ss, wrongly read as hh:mm
        corrected_time = pd.to_datetime(f"{time_val.minute}:{time_val.second}", format='%M:%S').time()
    else:
        # Correct time format
        try:
            corrected_time = time_val.time()
        except:
            import pdb; pdb.set_trace()
    return corrected_time