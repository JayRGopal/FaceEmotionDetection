def calculate_pain_expressivity(df):
    # Calculate fac_paiintsoft for each frame
    soft_columns = ["AU04_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU20_r", "AU26_r"]
    df['fac_paiintsoft'] = df[soft_columns].mean(axis=1) / 5
    
    # Calculate fac_paiinthard for each frame
    hard_columns = ["AU04_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c", "AU20_c", "AU26_c"]
    df['fac_paiinthard'] = df[hard_columns].apply(lambda row: 0 if 0 in row.values else row['fac_paiintsoft'], axis=1)
    
    # Calculate overall features
    results = {
        'fac_paiintsoft_pct': (df[hard_columns] > 0).any(axis=1).mean(),
        'fac_paiintsoft_mean': df['fac_paiintsoft'].mean(),
        'fac_paiintsoft_std': df['fac_paiintsoft'].std(),
        'fac_paiinthard_mean': df['fac_paiinthard'].mean(),
        'fac_paiinthard_std': df['fac_paiinthard'].std()
    }

    # Ensure no NaNs - replace NaNs with 0 for aggregation metrics
    results = {k: 0 if pd.isna(v) else v for k, v in results.items()}

    # Return results as a DataFrame
    return pd.DataFrame([results])