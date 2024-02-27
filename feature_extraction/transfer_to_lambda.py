elements = [
    "eye_lmk_x_0", "eye_lmk_x_1", "eye_lmk_x_2", "eye_lmk_x_3", "eye_lmk_x_4", "eye_lmk_x_5", "eye_lmk_x_6", "eye_lmk_x_7",
    "eye_lmk_x_8", "eye_lmk_x_9", "eye_lmk_x_10", "eye_lmk_x_11", "eye_lmk_x_12", "eye_lmk_x_13", "eye_lmk_x_14", "eye_lmk_x_15",
    "eye_lmk_x_16", "eye_lmk_x_17", "eye_lmk_x_18", "eye_lmk_x_19", "eye_lmk_x_20", "eye_lmk_x_21", "eye_lmk_x_22", "eye_lmk_x_23",
    "eye_lmk_x_24", "eye_lmk_x_25", "eye_lmk_x_26", "eye_lmk_x_27", "eye_lmk_x_28", "eye_lmk_x_29", "eye_lmk_x_30", "eye_lmk_x_31",
    "eye_lmk_x_32", "eye_lmk_x_33", "eye_lmk_x_34", "eye_lmk_x_35", "eye_lmk_x_36", "eye_lmk_x_37", "eye_lmk_x_38", "eye_lmk_x_39",
    "eye_lmk_x_40", "eye_lmk_x_41", "eye_lmk_x_42", "eye_lmk_x_43", "eye_lmk_x_44", "eye_lmk_x_45", "eye_lmk_x_46", "eye_lmk_x_47",
    "eye_lmk_x_48", "eye_lmk_x_49", "eye_lmk_x_50", "eye_lmk_x_51", "eye_lmk_x_52", "eye_lmk_x_53", "eye_lmk_x_54", "eye_lmk_x_55",
    "eye_lmk_y_0", "eye_lmk_y_1", "eye_lmk_y_2", "eye_lmk_y_3", "eye_lmk_y_4", "eye_lmk_y_5", "eye_lmk_y_6", "eye_lmk_y_7",
    "eye_lmk_y_8", "eye_lmk_y_9", "eye_lmk_y_10", "eye_lmk_y_11", "eye_lmk_y_12", "eye_lmk_y_13", "eye_lmk_y_14", "eye_lmk_y_15",
    "eye_lmk_y_16", "eye_lmk_y_17", "eye_lmk_y_18", "eye_lmk_y_19", "eye_lmk_y_20", "eye_lmk_y_21", "eye_lmk_y_22", "eye_lmk_y_23",
    "eye_lmk_y_24", "eye_lmk_y_25", "eye_lmk_y_26", "eye_lmk_y_27", "eye_lmk_y_28", "eye_lmk_y_29", "eye_lmk_y_30", "eye_lmk_y_31",
    "eye_lmk_y_32", "eye_lmk_y_33", "eye_lmk_y_34", "eye_lmk_y_35", "eye_lmk_y_36", "eye_lmk_y_37", "eye_lmk_y_38", "eye_lmk_y_39",
    "eye_lmk_y_40", "eye_lmk_y_41", "eye_lmk_y_42", "eye_lmk_y_43", "eye_lmk_y_44", "eye_lmk_y_45", "eye_lmk_y_46", "eye_lmk_y_47",
    "eye_lmk_y_48", "eye_lmk_y_49", "eye_lmk_y_50", "eye_lmk_y_51", "eye_lmk_y_52", "eye_lmk_y_53", "eye_lmk_y_54", "eye_lmk_y_55"
]

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

def compute_ear(row):
    # Right eye
    d1 = euclidean((row['eye_lmk_x_10'], row['eye_lmk_y_10']), (row['eye_lmk_x_18'], row['eye_lmk_y_18']))
    d2 = euclidean((row['eye_lmk_x_12'], row['eye_lmk_y_12']), (row['eye_lmk_x_16'], row['eye_lmk_y_16']))
    d3 = euclidean((row['eye_lmk_x_8'], row['eye_lmk_y_8']), (row['eye_lmk_x_14'], row['eye_lmk_y_14']))
    right_ear = (d1 + d2) / (2.0 * d3)
    
    # Left eye
    d4 = euclidean((row['eye_lmk_x_38'], row['eye_lmk_y_38']), (row['eye_lmk_x_46'], row['eye_lmk_y_46']))
    d5 = euclidean((row['eye_lmk_x_40'], row['eye_lmk_y_40']), (row['eye_lmk_x_44'], row['eye_lmk_y_44']))
    d6 = euclidean((row['eye_lmk_x_36'], row['eye_lmk_y_36']), (row['eye_lmk_x_42'], row['eye_lmk_y_42']))
    left_ear = (d4 + d5) / (2.0 * d6)
    
    # Overall EAR
    return (right_ear + left_ear) / 2.0

def process_video_df(df):
    # Calculate EAR for each frame
    df['EAR'] = df.apply(compute_ear, axis=1)
    
    # Identify blinks
    df['blink'] = (df['EAR'] < 0.2) & (df['EAR'].shift(1) >= 0.2)
    df['mov_blinkframe'] = df.index[df['blink']].tolist()
    df['mov_blink_ear'] = df.loc[df['blink'], 'EAR']
    
    # Identify blink frames
    df['is_blink'] = (df['EAR'] < 0.2) & (df['EAR'].shift(1) >= 0.2)
    blinks = df[df['is_blink']].copy()
    
    # Calculate blink-related features
    if not blinks.empty:
        blinks['mov_blinkdur'] = blinks['timestamp'].diff().fillna(0)
        features = {
            'mov_blink_ear_mean': blinks['EAR'].mean(),
            'mov_blink_ear_std': blinks['EAR'].std(),
            'mov_blink_count': blinks['is_blink'].sum(),
            'mov_blinkdur_mean': blinks['mov_blinkdur'].mean(),
            'mov_blinkdur_std': blinks['mov_blinkdur'].std(),
        }
    else:
        # Handle case with no blinks
        features = {
            'mov_blink_ear_mean': 0,
            'mov_blink_ear_std': 0,
            'mov_blink_count': 0,
            'mov_blinkdur_mean': 0,
            'mov_blinkdur_std': 0,
        }

    
    return pd.DataFrame([features])