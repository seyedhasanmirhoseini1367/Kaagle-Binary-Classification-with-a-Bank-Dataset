
import scipy.io
from matplotlib import pyplot as plt

# Load the .mat file
raw = scipy.io.loadmat("C:/Users\HP\Downloads\Compressed\Interference Testing.mat")

raw = scipy.io.loadmat("C:/Users\HP\Downloads\Compressed\Primary Measurements.mat")

data = raw['data'][:,1]
data.shape


n = len(data) // 60

plt.plot(range(0,n), data[:n])
plt.xlabel('Samples')
plt.ylabel(' Sample Frequency')
plt.show()

import numpy as np

# Extract the inter-sample interval (isi) in milliseconds
isi_ms = raw['isi'][0][0]  # Value is 5 ms

# Convert isi to seconds
isi_seconds = isi_ms / 1000

# Calculate the sampling rate
sampling_rate = 1 / isi_seconds

print(f"Sampling Rate: {sampling_rate} Hz")