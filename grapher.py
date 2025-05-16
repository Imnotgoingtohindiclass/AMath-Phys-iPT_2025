import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Load the audio file
sample_rate, data = wavfile.read('2.wav')  # Replace with your file path

# If stereo, convert to mono
if len(data.shape) == 2:
    data = data.mean(axis=1)

# Generate the spectrogram
frequencies, times, Sxx = spectrogram(data, sample_rate)

# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
plt.title('Spectrogram of Audio File')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()
