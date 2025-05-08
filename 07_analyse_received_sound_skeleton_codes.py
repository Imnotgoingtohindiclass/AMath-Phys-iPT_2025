# INITIALIZATION
import numpy as np
import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# FUNCTION DEFINITIONS

def samp_index_to_t(n, samp_rate):
    return n * 1.0 / samp_rate

def samp_ts(samps, samp_rate):
    ns = np.array(range(0, len(samps)))
    return samp_index_to_t(ns, samp_rate)

def sound_wave_model(ts):
    A = 673
    f = 1319
    t_max = 2.135547
    c = -6511
    return A * np.cos(2 * math.pi * f * (ts - t_max)) + c

# EXECUTION

sound_data = read('02_sound.wav')

# Sampling rate in samples per second
samp_rate = sound_data[0]
print(f"Sample rate: {samp_rate} Hz")

# Sound samples
samps = sound_data[1]

# Time values for each sample
ts = samp_ts(samps, samp_rate)

# Determine plot padding duration
padding = 3.0
end_time = ts[-1] + padding

# GRAPH PLOTTING

# Figure 1 – Full envelope with 3s padding
plt.clf()
plt.plot(ts, samps)
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([0, end_time])
plt.ylim([-7500, -5500])
plt.title('Full Received Sound Envelope (+3s Padding)')
plt.savefig('08 Received sound envelope Padded.png')

# Figure 2 – Zoomed in view
plt.figure()
plt.plot(ts, samps)
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([2.13, 2.14])
plt.ylim([-7500, -5500])
plt.title('Zoomed In: Signal Period')
plt.savefig('09a Received sound period - Raw.png')

# Figure 3 – Model vs actual
plt.figure()
plt.plot(ts, samps, label="Actual Signal")
plt.plot(ts, sound_wave_model(ts), "--", label="Model")
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([2.13, 2.14])
plt.ylim([-7500, -5500])
plt.title('Signal vs Mathematical Model')
plt.legend()
plt.savefig('09b Received sound period - Model.png')

plt.show()
print('Done!')
