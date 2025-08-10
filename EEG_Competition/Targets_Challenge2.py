import pandas as pd

'''
Target Variables (validation only):

P-factor: General psychopathology factor (continuous score)
Internalizing: Inward-focused traits dimension (continuous score)
Externalizing: Outward-directed traits dimension (continuous score)
Attention: Focus and distractibility dimension (continuous score)
'''


def target_challenge2(df, participant_id):
    target_cols = ['p_factor', 'attention', 'internalizing', 'externalizing']

    # Filter the dataframe for participant
    participant_data = df[df['participant_id'] == participant_id]

    if participant_data.empty:
        raise ValueError(f"No data found for participant_id: {participant_id}")

    targets = []
    for col in target_cols:
        values = participant_data[col].values
        targets.append(float(values[0]))

    return targets


#tar = target_challenge2(ptd, 'sub-NDARFW972KFQ')
#print(tar)  # Output: [-0.393, 1.491, -0.565, 0.328]