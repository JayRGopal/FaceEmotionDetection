def correct_time_format(time_str):
    time_str = time_str.strip()
    
    # Try parsing the time as "HH:MM:SS" first
    time_val = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
    if pd.isna(time_val):
        # If parsing failed, try "M:SS" or "MM:SS" next
        time_val = pd.to_datetime(time_str, format='%M:%S', errors='coerce')
        if pd.isna(time_val):
            # If still fails, handle or raise error
            raise ValueError("Invalid time format")
        else:
            # Adjust for "M:SS" or "MM:SS" as "00:M:SS"
            corrected_time = pd.to_datetime(f"00:{time_str}", format='%H:%M:%S').time()
    else:
        corrected_time = time_val.time()

    return corrected_time
