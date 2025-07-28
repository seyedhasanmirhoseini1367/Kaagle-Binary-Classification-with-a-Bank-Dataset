"""
This script demonstrates how to load and display basic information from an EEGLAB .set
file containing electroencephalography (EEG) data using the MNE-Python library.

The loaded dataset pertains to 'sub-NDARFW972KFQ', likely a participant from a study,
and was recorded during a 'DiaryOfAWimpyKid' task.

Key characteristics of the loaded EEG data:
- Number of Channels: 129
- Sampling Frequency: 500.0 Hz (meaning 500 data points per second per channel)
- Duration: Approximately 118.93 seconds of continuous recording
- Lowpass Filter: 250.0 Hz (indicating frequencies above 250 Hz were attenuated during acquisition)
- Highpass Filter: 0.0 Hz (suggesting no high-pass filtering was applied during acquisition, or it was at DC)

This initial loading step is fundamental for further EEG signal processing and analysis.
"""

# pip install mne
import mne.io as io

file_path = 'C:/Users\HP\Downloads\Compressed\sub-NDARFW972KFQ\eeg\sub-NDARFW972KFQ_task-DiaryOfAWimpyKid_eeg.set'  # Replace with the actual path to your file

raw = io.read_raw_eeglab(file_path, preload=True)
raw.info

'''
<Info | 8 non-empty values
 bads: []
 ch_names: E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, ...
 chs: 129 EEG
 custom_ref_applied: False
 dig: 132 items (3 Cardinal, 129 EEG)
 highpass: 0.0 Hz
 lowpass: 250.0 Hz
 meas_date: unspecified
 nchan: 129
 projs: []
 sfreq: 500.0 Hz
>

'''

print(f"Number of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

