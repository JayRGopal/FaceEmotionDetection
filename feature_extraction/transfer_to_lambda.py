def add_time_strings(t1, t2):
    total_seconds = sum(x.total_seconds() for x in [pd.to_timedelta(t) for t in [t1, t2]])
    return str(pd.to_timedelta(total_seconds, unit='s'))
