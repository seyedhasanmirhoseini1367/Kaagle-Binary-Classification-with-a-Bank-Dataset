
"""
This script loads a Tab-Separated Values (.tsv) file containing
detailed information about the EEG channels for a specific recording
(sub-NDARFW972KFQ, task-contrastChangeDetection, run-2).

The 'channels.tsv' file provides metadata such as:
- 'name': The name/label of each individual EEG channel (e.g., E1, Cz).
- 'type': The type of sensor (e.g., EEG).
- 'units': The units in which the data for that channel is measured (e.g., uV for microvolts).

"""

import pandas as pd

file_path = 'C:/Users/HP/Downloads/Compressed/sub-NDARFW972KFQ/eeg/sub-NDARFW972KFQ_task-contrastChangeDetection_run-2_channels.tsv'

# Use read_csv and specify the tab delimiter
df_channels = pd.read_csv(file_path, sep='\t')

'''
df_channels
Out[90]: 
     name type units
0      E1  EEG    uV
1      E2  EEG    uV
2      E3  EEG    uV
3      E4  EEG    uV
4      E5  EEG    uV
..    ...  ...   ...
124  E125  EEG    uV
125  E126  EEG    uV
126  E127  EEG    uV
127  E128  EEG    uV
128    Cz  EEG    uV
[129 rows x 3 columns]
'''
