# pip install mne
import mne.io as io

file_path = 'C:/Users\HP\Downloads\Compressed\sub-NDARFW972KFQ\eeg\sub-NDARFW972KFQ_task-DiaryOfAWimpyKid_eeg.set'  # Replace with the actual path to your file
file_path = "C:/Users\HP\Downloads\Compressed\sub-NDARFW972KFQ\eeg\sub-NDARFW972KFQ_task-contrastChangeDetection_run-3_eeg.set"  # Replace with the actual path to your file

raw = io.read_raw_eeglab(file_path, preload=True)
raw.info
chan_names = raw.ch_names
Sampling_frequency = int(raw.info['sfreq'])

import mne.io as io
import matplotlib.pyplot as plt

print(f"Successfully loaded {file_path}")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

# Access the data (e.g., first 5 channels, first 2 seconds)
data, times = raw[:5, :Sampling_frequency]
print(f"Shape of data (first 5 channels, 2 seconds): {data.shape}")

# Plot the raw data (first few channels)
raw.plot(n_channels=5, scalings='auto', title='Raw EEG Data')
plt.show()

# Plot the power spectral density
raw.plot_psd(fmax=50, average=True, spatial_colors=False)
plt.show()
