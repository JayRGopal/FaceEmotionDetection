def merge_dicts(dict1, dict2):
    return {
        key: pd.concat([dict1[key], dict2[key].drop(columns=['Datetime'], errors='ignore')], axis=1) 
        for key in dict1
    }