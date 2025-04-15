# INITIALIZATION

# import os

# import pip
# pip.main(['install','numpy'])
import numpy as np
import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# FUNCTION DEFINITIONS

def samp_index_to_t(n, samp_rate):
    t = n * 1.0 / samp_rate
    return t

def samp_ts(samps, samp_rate):
    ns = np.array(range(0, len(samps)))
    ts = samp_index_to_t(ns, samp_rate)
    return ts

def sound_wave_model(ts):
    # A sound wave is mathematically modelled using a cosine function.
    # Returns an array of samples of this mathematical model.
    A = 673
    f = 1319
    t_max = 2.135547
    c = -6511
    model_samps = A * np.cos(2 * math.pi * f * (ts - t_max)) + c
    return model_samps

# EXECUTION

sound_data = read('06_received_sound.wav')

# Sampling rate in samples per second
samp_rate = sound_data[0]
print(samp_rate)

# Sound samples
samps = sound_data[1]

# Times, in seconds, of the signal values in samps
ts = samp_ts(samps, samp_rate)

# GRAPH PLOTTING

# The graph scales can be adjusted to see different features of the graph.

# Figure 1
# The axes scales are set to see the envelope representing the pulses
# of the morse code, with the unwanted durations before and after
# the signal cropped away. Note: The signal is noisy. Ignore the noise.
plt.clf()
x_axis = ts
y_axis = samps
plt.plot(x_axis, y_axis)
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([0.3, 8.1])
plt.ylim([-7500, -5500])
plt.savefig('08 Received sound envelope.png')

# Figure 2
# The axes scales are set to see the periodicity of the sound signal 
# of a '1'/on-pulse with respect to time.
# Hover the cursor over a peak (e.g. maximum positive displacement) on the graph.
# Record the x-coordinate (t / s) shown on the bottom-right of the window.
# Repeat by hovering the cursor over the next peak (1 period later).
# Calculate sound_period = the difference between the two values of t/s.
# Calculate sound_freq = 1/sound_period.
# (sound_freq can also be preliminarily estimated with a music tuner app.)
# Note: There may be overtones of frequencies that are integer multiples
# of the actual sound_freq.
plt.figure()
x_axis = ts
y_axis = samps
plt.plot(x_axis, y_axis)
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([2.13, 2.14])
plt.ylim([-7500, -5500])
plt.savefig('09a Received sound period - Raw.png')

# Figure 3
# Mathematical model to fit the data in Figure 3
plt.figure()
x_axis = ts
y_axis_a = samps
y_axis_b = sound_wave_model(ts)
plt.plot(x_axis, y_axis_a, x_axis, y_axis_b, "--")
plt.xlabel('time / s')
plt.ylabel('signal / arbitrary units')
plt.xlim([2.13, 2.14])
plt.ylim([-7500, -5500])
plt.savefig('09b Received sound period - Model.png')

plt.show()

print('Done!')
