import numpy as np
import pandas as pd

'''
Targets
Reaction Time: The time difference (in seconds) between the stimulus onset (left_target or right_target) and the participant's response (right_buttonPress or left_buttonPress).
Hit Accuracy: A binary classification target indicating whether the participant responded correctly (smiley_face) or not (sad_face) during each trial.
'''
    
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
            accuracy = press_buttons.iloc[0]['feedback']
            rt_target.append((np.round(rt, 3), accuracy))

    return rt_target



'''
ccd_run1_path = "D:\EEGChallenge\hbn_bids_R1\sub-NDARAC904DMU\eeg\sub-NDARAC904DMU_task-contrastChangeDetection_run-1_events.tsv"
df1 = pd.read_csv(ccd_run1_path, sep='\t')

rt1 = reaction_time(df1)
print(rt1)

[(2.13, 'smiley_face'),
 (1.96, 'smiley_face'),
 (2.02, 'smiley_face'),
 (1.72, 'smiley_face'),
 (1.8, 'smiley_face'),
 (1.72, 'smiley_face'),
 (1.842, 'smiley_face'),
 (1.5, 'smiley_face'),
 (2.33, 'smiley_face'),
 (1.77, 'smiley_face'),
 (1.37, 'smiley_face'),
 (1.64, 'smiley_face'),
 (1.35, 'smiley_face'),
 (1.98, 'smiley_face'),
 (1.82, 'smiley_face'),
 (2.18, 'smiley_face'),
 (1.382, 'smiley_face'),
 (1.95, 'smiley_face'),
 (1.64, 'smiley_face'),
 (1.52, 'smiley_face'),
 (1.69, 'smiley_face'),
 (1.22, 'smiley_face')]
'''
