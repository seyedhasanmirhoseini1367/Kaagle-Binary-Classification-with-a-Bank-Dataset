import numpy as np
import pandas as pd


def reaction_time(df):
    rt_target = []
    # Find indices where 'feedback' contains 'face' (smiley_face or sad_face)
    target_indices = df.index[df['feedback'].str.contains('face', na=False)].tolist()

    for idx in target_indices:
        # Check if the current row's 'value' ends with 'buttonPress'
        press_buttons = df[(df.index == idx) & (df['value'].str.contains('buttonPress'))]

        # Check if the previous row's 'value' is a target event (left_target or right_target)
        prev_value = df.loc[idx - 1, 'value']
        if prev_value not in ['left_target', 'right_target']:
            continue

        if not press_buttons.empty:
            rt = press_buttons.iloc[0]['onset'] - df.loc[idx-1, 'onset']
            target = press_buttons.iloc[0]['feedback']
            rt_target.append((np.round(rt, 3), target))

    return rt_target
