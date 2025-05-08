import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

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

# Reverse the MORSE_CODE_DICT to decode Morse code pulses into characters
REVERSE_MORSE_CODE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

def audio_to_pulses(audio, samp_rate, tone_freq, threshold=0.1):
    # Convert audio samples into pulses of '1's (tone) and '0's (silence)
    duration = len(audio) / samp_rate
    ts = samp_ts(0, duration, samp_rate)
    angular_freq = 2 * np.pi * tone_freq
    sine_wave = np.sin(angular_freq * ts)
    normalized_audio = audio / float(np.max(np.abs(audio)))  # Normalize the audio
    correlation = np.correlate(normalized_audio, sine_wave, mode='same')
    pulses = ''.join(['1' if abs(corr) > threshold else '0' for corr in correlation])
    return pulses

def pulses_to_msg(pulses):
    # Convert pulses of '1's and '0's into a decoded message string
    words = pulses.split('0000000')  # Split words by 7-dot silence
    decoded_words = []
    for word in words:
        letters = word.split('000')  # Split letters by 3-dot silence
        decoded_letters = []
        for letter in letters:
            morse = letter.replace('111', '-').replace('1', '.').replace('0', '')  # Convert pulses to Morse code
            if morse in REVERSE_MORSE_CODE_DICT:
                decoded_letters.append(REVERSE_MORSE_CODE_DICT[morse])  # Decode Morse code to character
        decoded_words.append(''.join(decoded_letters))
    return ' '.join(decoded_words)

# Read the audio file
samp_rate, audio_data = read('02_sound.wav')

# Convert the audio data into pulses
pulses = audio_to_pulses(audio_data, samp_rate, tone_freq)
print(f"Decoded pulses: {pulses}")

# Convert the pulses into the original message
decoded_message = pulses_to_msg(pulses)
print(f"Decoded message: {decoded_message}")

# Plot the original waveform
duration = len(audio_data) / samp_rate
ts = samp_ts(0, duration, samp_rate)

plt.figure()
plt.plot(ts[:len(audio_data)], audio_data)  # Plot the waveform of the audio signal
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('Decoded Morse Code Signal')
plt.show()