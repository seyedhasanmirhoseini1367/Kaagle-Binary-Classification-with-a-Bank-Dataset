"""
This script loads a JSON (.json) file that serves as a "sidecar" metadata file
for an EEG recording, following the Brain Imaging Data Structure (BIDS) standard.

For the specific recording (sub-NDARFW972KFQ, task-FunwithFractals),
this '_eeg.json' file contains crucial metadata about the EEG acquisition,
which can include details such as:
- PowerLineFrequency: 60
- TaskName: FunwithFractals
- EEGChannelCount: 129
- EEGReference: Cz
- RecordingType: continuous
- RecordingDuration: 165.804
- SamplingFrequency: 500
- SoftwareFilters: n/a

"""

import json

file_path = 'C:/Users/HP/Downloads/Compressed/sub-NDARFW972KFQ/eeg/sub-NDARFW972KFQ_task-FunwithFractals_eeg.json'

try:
    with open(file_path, 'r') as f:
        eeg_metadata = json.load(f)

    print("Successfully loaded the EEG metadata JSON file:")
    for key, value in eeg_metadata.items():
        print(f"- {key}: {value}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from '{file_path}'. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")