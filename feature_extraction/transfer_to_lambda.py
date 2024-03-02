
elements = [f"eye_lmk_x_{i}" for i in range(56)] + [f"eye_lmk_y_{i}" for i in range(56)] + [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]

def calculate_landmark_displacement(df):
    # Preparing the column names for X, Y, Z coordinates
    x_cols = [f'X_{i}' for i in range(68)]
    y_cols = [f'Y_{i}' for i in range(68)]
    z_cols = [f'Z_{i}' for i in range(68)]
    
    # Calculating the displacement for each landmark across frames
    disp_cols = []
    for x_col, y_col, z_col in zip(x_cols, y_cols, z_cols):
        disp_col = f'{x_col}_disp'
        df[disp_col] = np.sqrt((df[x_col].diff() ** 2) + (df[y_col].diff() ** 2) + (df[z_col].diff() ** 2))
        disp_cols.append(disp_col)
    
    # Calculating the mean and standard deviation of displacements for each landmark
    output_df = pd.DataFrame()
    for col in disp_cols:
        landmark_num = col.split('_')[1]
        output_df[f'fac_lmk{landmark_num}disp_mean'] = [df[col].mean()]
        output_df[f'fac_lmk{landmark_num}disp_std'] = [df[col].std()]
    
    # Return a DataFrame with calculated mean and standard deviation for each landmark displacement
    return output_df