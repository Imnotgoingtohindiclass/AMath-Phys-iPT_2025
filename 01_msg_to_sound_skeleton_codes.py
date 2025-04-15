import numpy as np
import math # who knows what we're importing ts for...
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# morse code dictionary mapping letters and numbers to their corresponding morse code representations
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----', ' ': ' '
}

# defining key parameters for the morse code audio generation
samp_rate = 44100  # sampling rate in hertz, determines the number of audio samples per second
pulse_duration = 0.1  # duration of a single dot in seconds
tone_freq = 700  # frequency of the tone in hertz
sound_amplitude = 32767  # maximum amplitude for a 16-bit integer wav format

def pulse_index_to_start_t(j, pulse_duration):
    # calculates the starting time of a given pulse based on its index and the duration of each pulse
    return j * pulse_duration

def t_to_samp_index(t, samp_rate):
    # converts a given time in seconds to a sample index based on the sampling rate
    return int(round(t * samp_rate))

def samp_ts(start_t, end_t, samp_rate):
    # generates an array of time values for each sample between a given start and end time
    return np.linspace(start_t, end_t, int((end_t - start_t) * samp_rate), endpoint=False)

def tone(tone_freq, sound_amplitude, start_t, end_t, samp_rate):
    # generates a sine wave tone of a given frequency and amplitude for the specified time range
    angular_freq = 2 * np.pi * tone_freq  # converts frequency to angular frequency
    ts = samp_ts(start_t, end_t, samp_rate)  # generate time steps for the given duration
    return (sound_amplitude * np.sin(angular_freq * ts)).astype(np.int16)  # compute sine wave and scale to int16

def msg_to_pulses(msg):
    # converts a message string into a string of '1's (tones) and '0's (silence) representing morse code pulses.
    pulses = []
    words = msg.upper().split(' ') #split the input message into words and convert to uppercase.

    for word in words: #loop through each word
        word_pulses = []
        for char in word: #loop through each character in the word.
            if char in MORSE_CODE_DICT: #check if the character is in the morse code dictionary.
                morse = MORSE_CODE_DICT[char] #get the morse code representation of the character.

                char_pulses_with_zeros = []
                for i, sym in enumerate(morse):
                    if i > 0:
                        char_pulses_with_zeros.append('0') # add a silence of three dot durations between each morse symbol.
                    if sym == '.':
                        char_pulses_with_zeros.append('1') # add a '1' for a dot (short tone).
                    else:
                        char_pulses_with_zeros.append('111') # add '111' for a dash (long tone).

                word_pulses.append(''.join(char_pulses_with_zeros))
        pulses.append('0'.join(word_pulses)) # add a silence of three dot durations between each letter.
    return '0'.join(pulses) # add a silence of three dot durations between each word.

def pulses_to_samps(pulses, pulse_duration, tone_freq, sound_amplitude, samp_rate):
    # converts a sequence of pulses into an audio sample array.
    samps = np.array([], dtype=np.int16)  # initialize an empty numpy array for storing the samples
    for pulse in pulses: #loop through each pulse ('1' or '0')
        start_t = len(samps) / samp_rate  # calculate the start time of the current pulse
        end_t = start_t + pulse_duration  # calculate the end time of the current pulse
        if pulse == '1':  # if the pulse represents a tone
            samps = np.concatenate((samps, tone(tone_freq, sound_amplitude, start_t, end_t, samp_rate))) #add a tone to the samples.
        else:  # if the pulse represents silence
            samps = np.concatenate((samps, np.zeros(int(pulse_duration * samp_rate), dtype=np.int16))) #add silence to the samples.
    return samps  # return the final sample array

# message to encode into morse code
msg = "SOS"
print(f"message: {msg}")

# convert the message into morse code pulses
pulses = msg_to_pulses(msg)
print(f"pulses: {pulses}")

# generate the sound samples based on the pulses
samps = pulses_to_samps(pulses, pulse_duration, tone_freq, sound_amplitude, samp_rate)

# generate an array of time values corresponding to the audio samples
ts = samp_ts(0, pulse_index_to_start_t(len(pulses), pulse_duration), samp_rate)

# save the generated audio samples as a wav file
write('morse_code.wav', samp_rate, samps)

# plot the waveform of the generated morse code signal
plt.figure()
plt.plot(ts[:len(samps)], samps)  # plot the waveform with time on x-axis and amplitude on y-axis
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('generated morse code signal')
plt.ylim([-32768, 32767])
plt.savefig('morse_waveform.png')  # save the plot as an image file
plt.show()

print("done!")