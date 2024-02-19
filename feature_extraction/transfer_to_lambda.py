def process_gaze_data(df):
    # Ensure all gaze-related columns are floats
    gaze_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
    df[gaze_cols] = df[gaze_cols].astype(float)
    
    # Initialize output dictionary
    output_dict = {}
    
    # Mapping for renaming
    rename_map = {
        'gaze_0_x': 'righteyex', 'gaze_0_y': 'righteyey', 'gaze_0_z': 'righteyez',
        'gaze_1_x': 'lefteyex', 'gaze_1_y': 'lefteyey', 'gaze_1_z': 'lefteyez'
    }
    
    # Calculate mean and std for each gaze direction component and rename
    for col in gaze_cols:
        new_base_name = rename_map[col]
        output_dict[f"mov_{new_base_name}_mean"] = df[col].mean()
        output_dict[f"mov_{new_base_name}_std"] = df[col].std()
    
    # Calculate Euclidean displacement for each eye in each frame
    df['mov_leyedisp'] = np.sqrt((df['gaze_1_x'].diff()**2 + df['gaze_1_y'].diff()**2 + df['gaze_1_z'].diff()**2).fillna(0))
    df['mov_reyedisp'] = np.sqrt((df['gaze_0_x'].diff()**2 + df['gaze_0_y'].diff()**2 + df['gaze_0_z'].diff()**2).fillna(0))
    
    # Add mean and std for the Euclidean displacements to output dict
    output_dict['mov_leyedisp_mean'] = df['mov_leyedisp'].mean()
    output_dict['mov_leyedisp_std'] = df['mov_leyedisp'].std()
    output_dict['mov_reyedisp_mean'] = df['mov_reyedisp'].mean()
    output_dict['mov_reyedisp_std'] = df['mov_reyedisp'].std()
    
    # Create output DataFrame from the output_dict
    output_df = pd.DataFrame([output_dict])
    
    return output_df


def process_head_movement(df):
    # Ensure the pose columns are floats
    pose_cols = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    df[pose_cols] = df[pose_cols].astype(float)
    
    # Calculate Euclidean head movement (displacement)
    df['mov_headvel'] = np.sqrt(df[['pose_Tx', 'pose_Ty', 'pose_Tz']].diff().fillna(0).pow(2).sum(axis=1))
    
    # Assign frame-wise pitch, yaw, and roll directly from pose_Rx, pose_Ry, pose_Rz
    df['mov_hposepitch'] = df['pose_Rx']
    df['mov_hposeyaw'] = df['pose_Ry']
    df['mov_hposeroll'] = df['pose_Rz']
    
    # Calculate angular head movement using diff for pose_Rx, pose_Ry, pose_Rz, then take Euclidean norm
    df['mov_hposedist'] = np.sqrt(df[['pose_Rx', 'pose_Ry', 'pose_Rz']].diff().fillna(0).pow(2).sum(axis=1))
    
    # Calculate mean and std for the new variables
    output_dict = {}
    variables = ['mov_headvel', 'mov_hposepitch', 'mov_hposeyaw', 'mov_hposeroll', 'mov_hposedist']
    for var in variables:
        output_dict[f"{var}_mean"] = df[var].mean()
        output_dict[f"{var}_std"] = df[var].std()
    
    # Create output DataFrame from the output_dict
    output_df = pd.DataFrame([output_dict])
    
    return output_df


